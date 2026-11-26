from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import mimetypes
import os
import secrets
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
DATA_DIR = Path(os.getenv("NB_DATA_DIR", str(ROOT / "data"))).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
MODEL_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "neurobridge.sqlite3"
MAX_UPLOAD_BYTES = int(os.getenv("NB_MAX_UPLOAD_MB", "500")) * 1024 * 1024
ACCESS_CODE = os.getenv("NB_ACCESS_CODE", "dev-only-change-me")
SECRET_KEY = os.getenv("NB_SECRET_KEY", "development-secret-change-before-deployment")
TOKEN_TTL = int(os.getenv("NB_TOKEN_TTL_SECONDS", "43200"))
ALLOWED_EXTENSIONS = {".edf", ".bdf", ".csv", ".tsv", ".txt", ".json", ".pdf", ".png", ".jpg", ".jpeg", ".webp"}

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="NeuroBridge EEG Workbench API", version="3.0.2")

# Optional CORS for a Netlify-hosted frontend calling this backend directly.
# Large EEG file uploads go browser -> backend directly; they do not pass through
# Netlify's proxy. Keep this allow-list restricted to the exact frontend origin.
CORS_ORIGINS = [x.strip().rstrip("/") for x in os.getenv("NB_CORS_ORIGINS", "").split(",") if x.strip()]
if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA journal_mode=WAL")
    return con


def init_db() -> None:
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS studies (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, subject_code TEXT, age INTEGER,
            sex TEXT, study_date TEXT, montage TEXT, sampling_rate REAL, context TEXT,
            source_text TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS files (
            id TEXT PRIMARY KEY, study_id TEXT REFERENCES studies(id) ON DELETE SET NULL,
            original_name TEXT NOT NULL, stored_name TEXT NOT NULL, extension TEXT NOT NULL,
            mime_type TEXT, size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL,
            uploaded_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'uploaded',
            error TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_files_study ON files(study_id);
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY, file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            study_id TEXT, created_at TEXT NOT NULL, kind TEXT NOT NULL,
            summary_json TEXT NOT NULL, extracted_text TEXT
        );
        CREATE TABLE IF NOT EXISTS study_pages (
            study_id TEXT NOT NULL REFERENCES studies(id) ON DELETE CASCADE,
            file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            page_num INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(study_id,file_id,page_num)
        );
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, model_path TEXT NOT NULL,
            metrics_json TEXT NOT NULL, feature_names_json TEXT NOT NULL,
            classes_json TEXT NOT NULL, dataset_revision INTEGER NOT NULL,
            training_files_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS app_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT OR IGNORE INTO app_meta(key,value) VALUES ('dataset_revision','0');
        """)


init_db()


class LoginBody(BaseModel):
    access_code: str = Field(min_length=1, max_length=256)


class StudyBody(BaseModel):
    id: str | None = None
    title: str = Field(default="Untitled EEG study", max_length=160)
    subject_code: str | None = Field(default=None, max_length=80)
    age: int | None = Field(default=None, ge=0, le=120)
    sex: str | None = Field(default=None, max_length=40)
    study_date: str | None = Field(default=None, max_length=40)
    montage: str | None = Field(default=None, max_length=160)
    sampling_rate: float | None = Field(default=None, gt=0, le=100000)
    context: str | None = Field(default=None, max_length=2000)
    source_text: str | None = Field(default=None, max_length=50000)


class PagePick(BaseModel):
    file_id: str
    pages: list[int]


class PagesBody(BaseModel):
    selections: list[PagePick]


def sign_token(expiry: int) -> str:
    raw = f"nb3:{expiry}:{secrets.token_urlsafe(12)}".encode()
    sig = hmac.new(SECRET_KEY.encode(), raw, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=") + "." + base64.urlsafe_b64encode(sig).decode().rstrip("=")


def verify_token(token: str) -> bool:
    try:
        a, b = token.split(".", 1)
        raw = base64.urlsafe_b64decode(a + "=" * (-len(a) % 4))
        sig = base64.urlsafe_b64decode(b + "=" * (-len(b) % 4))
        expected = hmac.new(SECRET_KEY.encode(), raw, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            return False
        parts = raw.decode().split(":")
        return len(parts) == 3 and parts[0] == "nb3" and int(parts[1]) > int(time.time())
    except Exception:
        return False


def require_auth(authorization: str | None = Header(default=None)) -> None:
    if not authorization or not authorization.startswith("Bearer ") or not verify_token(authorization[7:]):
        raise HTTPException(status_code=401, detail="Access code session expired or invalid. Sign in again.")


def row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row else None


def bump_dataset_revision() -> int:
    with db() as con:
        con.execute("UPDATE app_meta SET value=CAST(value AS INTEGER)+1 WHERE key='dataset_revision'")
        return int(con.execute("SELECT value FROM app_meta WHERE key='dataset_revision'").fetchone()[0])


def current_revision() -> int:
    with db() as con:
        return int(con.execute("SELECT value FROM app_meta WHERE key='dataset_revision'").fetchone()[0])


@app.get("/api/health")
def health():
    return {"ok": True, "service": "NeuroBridge API", "version": "3.0.0", "storage": "persistent volume required for cloud hosting"}


@app.post("/api/auth/login")
def login(body: LoginBody):
    if not hmac.compare_digest(body.access_code.encode(), ACCESS_CODE.encode()):
        raise HTTPException(status_code=401, detail="Incorrect access code.")
    return {"token": sign_token(int(time.time()) + TOKEN_TTL), "expires_in": TOKEN_TTL}


@app.get("/api/stats")
def stats(_: None = Depends(require_auth)):
    with db() as con:
        file_count = con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        study_count = con.execute("SELECT COUNT(*) FROM studies").fetchone()[0]
        total_bytes = con.execute("SELECT COALESCE(SUM(size_bytes),0) FROM files").fetchone()[0]
        analysis_count = con.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        model_count = con.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    return {"files": file_count, "studies": study_count, "bytes_stored": total_bytes,
            "analyses": analysis_count, "models": model_count, "dataset_revision": current_revision()}


@app.get("/api/studies")
def list_studies(_: None = Depends(require_auth)):
    with db() as con:
        rows = con.execute("SELECT * FROM studies ORDER BY updated_at DESC").fetchall()
    return [row_dict(r) for r in rows]


@app.post("/api/studies")
def save_study(body: StudyBody, _: None = Depends(require_auth)):
    sid = body.id or str(uuid.uuid4())
    ts = now_iso()
    with db() as con:
        existing = con.execute("SELECT id FROM studies WHERE id=?", (sid,)).fetchone()
        values = (body.title, body.subject_code, body.age, body.sex, body.study_date,
                  body.montage, body.sampling_rate, body.context, body.source_text, ts)
        if existing:
            con.execute("""UPDATE studies SET title=?,subject_code=?,age=?,sex=?,study_date=?,montage=?,
                       sampling_rate=?,context=?,source_text=?,updated_at=? WHERE id=?""", values + (sid,))
        else:
            con.execute("""INSERT INTO studies(id,title,subject_code,age,sex,study_date,montage,sampling_rate,
                       context,source_text,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                       (sid,) + values[:-1] + (ts, ts))
        return row_dict(con.execute("SELECT * FROM studies WHERE id=?", (sid,)).fetchone())


@app.get("/api/studies/{study_id}")
def get_study(study_id: str, _: None = Depends(require_auth)):
    with db() as con:
        r = con.execute("SELECT * FROM studies WHERE id=?", (study_id,)).fetchone()
    if not r:
        raise HTTPException(404, "Study not found")
    return row_dict(r)


@app.post("/api/studies/{study_id}/attach-unassigned")
def attach_unassigned_files(study_id: str, _: None = Depends(require_auth)):
    with db() as con:
        if not con.execute("SELECT 1 FROM studies WHERE id=?", (study_id,)).fetchone():
            raise HTTPException(404, "Study not found")
        cur = con.execute("UPDATE files SET study_id=? WHERE study_id IS NULL", (study_id,))
        return {"attached": cur.rowcount}


@app.get("/api/studies/{study_id}/pages")
def get_selected_pages(study_id: str, _: None = Depends(require_auth)):
    with db() as con:
        if not con.execute("SELECT 1 FROM studies WHERE id=?", (study_id,)).fetchone():
            raise HTTPException(404, "Study not found")
        rows = con.execute("SELECT p.file_id,p.page_num,f.original_name FROM study_pages p JOIN files f ON f.id=p.file_id WHERE p.study_id=? ORDER BY f.original_name,p.page_num", (study_id,)).fetchall()
    return [dict(r) for r in rows]


@app.put("/api/studies/{study_id}/pages")
def save_selected_pages(study_id: str, body: PagesBody, _: None = Depends(require_auth)):
    total = sum(len(x.pages) for x in body.selections)
    if total > 8:
        raise HTTPException(400, "Select a maximum of 8 PDF pages per study")
    validated = []
    for pick in body.selections:
        rec = get_file_record(pick.file_id)
        if rec["extension"] == ".pdf":
            doc = pdf_document(rec["path"])
            count = len(doc)
            doc.close()
            if any(p < 1 or p > count for p in pick.pages):
                raise HTTPException(400, f"Page number out of range for {rec['original_name']}")
        elif rec["extension"] in {".png", ".jpg", ".jpeg", ".webp"}:
            if any(p != 0 for p in pick.pages):
                raise HTTPException(400, "Uploaded images use page number 0 in the saved selection")
        else:
            raise HTTPException(415, f"{rec['original_name']} is not a PDF or image")
        validated.append((rec, sorted(set(pick.pages))))
    with db() as con:
        if not con.execute("SELECT 1 FROM studies WHERE id=?", (study_id,)).fetchone():
            raise HTTPException(404, "Study not found")
        con.execute("DELETE FROM study_pages WHERE study_id=?", (study_id,))
        for rec, pages in validated:
            if rec["study_id"] not in (None, study_id):
                raise HTTPException(409, "A selected PDF belongs to a different study")
            if rec["study_id"] is None:
                con.execute("UPDATE files SET study_id=? WHERE id=?", (study_id, rec["id"]))
            for page_num in pages:
                con.execute("INSERT INTO study_pages(study_id,file_id,page_num,created_at) VALUES(?,?,?,?)",
                            (study_id, rec["id"], page_num, now_iso()))
    return {"saved_pages": total}


@app.post("/api/files")
async def upload_file(file: UploadFile = File(...), study_id: str | None = Form(default=None), _: None = Depends(require_auth)):
    original = Path(file.filename or "upload").name
    ext = Path(original).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type: {ext or 'no extension'}")
    if study_id:
        with db() as con:
            if not con.execute("SELECT 1 FROM studies WHERE id=?", (study_id,)).fetchone():
                raise HTTPException(404, "Selected study was not found")
    file_id = str(uuid.uuid4())
    stored_name = file_id + ext
    target = UPLOAD_DIR / stored_name
    digest = hashlib.sha256()
    size = 0
    try:
        with target.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_BYTES // (1024*1024)} MB server limit")
                digest.update(chunk)
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "The selected file is empty")
        mime = file.content_type or mimetypes.guess_type(original)[0] or "application/octet-stream"
        with db() as con:
            con.execute("""INSERT INTO files(id,study_id,original_name,stored_name,extension,mime_type,
                       size_bytes,sha256,uploaded_at,status) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                       (file_id, study_id, original, stored_name, ext, mime, size, digest.hexdigest(), now_iso(), "uploaded"))
        revision = bump_dataset_revision() if ext in {".csv", ".tsv", ".txt"} else current_revision()
        return {"id": file_id, "study_id": study_id, "original_name": original, "size_bytes": size,
                "sha256": digest.hexdigest(), "status": "uploaded", "dataset_revision": revision}
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@app.get("/api/files")
def list_files(study_id: str | None = None, _: None = Depends(require_auth)):
    with db() as con:
        if study_id:
            rows = con.execute("SELECT * FROM files WHERE study_id=? ORDER BY uploaded_at DESC", (study_id,)).fetchall()
        else:
            rows = con.execute("SELECT * FROM files ORDER BY uploaded_at DESC LIMIT 300").fetchall()
    return [row_dict(r) for r in rows]


def get_file_record(file_id: str) -> dict[str, Any]:
    with db() as con:
        rec = row_dict(con.execute("SELECT * FROM files WHERE id=?", (file_id,)).fetchone())
    if not rec:
        raise HTTPException(404, "File not found")
    path = UPLOAD_DIR / rec["stored_name"]
    if not path.exists():
        raise HTTPException(410, "Stored file is missing from the server volume")
    rec["path"] = path
    return rec


@app.get("/api/files/{file_id}/download")
def download_file(file_id: str, _: None = Depends(require_auth)):
    rec = get_file_record(file_id)
    return FileResponse(rec["path"], filename=rec["original_name"], media_type=rec["mime_type"])


@app.delete("/api/files/{file_id}")
def delete_file(file_id: str, _: None = Depends(require_auth)):
    rec = get_file_record(file_id)
    with db() as con:
        con.execute("DELETE FROM files WHERE id=?", (file_id,))
    rec["path"].unlink(missing_ok=True)
    return {"deleted": True}


def pdf_document(path: Path):
    try:
        import fitz
        return fitz.open(path)
    except Exception as e:
        raise HTTPException(422, f"Unable to open PDF: {e}")


@app.get("/api/files/{file_id}/pdf-info")
def pdf_info(file_id: str, _: None = Depends(require_auth)):
    rec = get_file_record(file_id)
    if rec["extension"] != ".pdf":
        raise HTTPException(415, "This file is not a PDF")
    doc = pdf_document(rec["path"])
    pages = [{"page": i + 1, "width": round(p.rect.width, 1), "height": round(p.rect.height, 1),
              "has_text": bool(p.get_text().strip())} for i, p in enumerate(doc)]
    doc.close()
    return {"file_id": file_id, "pages": pages, "page_count": len(pages)}


@app.get("/api/files/{file_id}/pdf-page/{page_num}")
def pdf_page(file_id: str, page_num: int, _: None = Depends(require_auth)):
    rec = get_file_record(file_id)
    if rec["extension"] != ".pdf":
        raise HTTPException(415, "This file is not a PDF")
    doc = pdf_document(rec["path"])
    if page_num < 1 or page_num > len(doc):
        doc.close()
        raise HTTPException(404, "PDF page number out of range")
    pix = doc[page_num - 1].get_pixmap(matrix=__import__("fitz").Matrix(1.4, 1.4), alpha=False)
    data = pix.tobytes("png")
    doc.close()
    return Response(data, media_type="image/png", headers={"Cache-Control": "no-store"})


def load_table(path: Path, ext: str):
    import pandas as pd
    sep = "\t" if ext == ".tsv" else None
    try:
        return pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip")
    except Exception as e:
        raise ValueError(f"Could not parse tabular EEG file: {e}")


def finite_channel_columns(df):
    excluded = {"time", "time_s", "timestamp", "sample", "sample_index", "index", "trial_id",
                "task_label", "label", "event", "condition", "utterance_text", "transcript"}
    cols = []
    for c in df.columns:
        if str(c).strip().lower() in excluded:
            continue
        vals = __import__("pandas").to_numeric(df[c], errors="coerce")
        if vals.notna().sum() >= max(3, int(len(df) * 0.5)):
            cols.append(c)
    return cols


def spectral_bands(x: np.ndarray, fs: float) -> dict[str, float]:
    from scipy.signal import welch
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 16 or not np.isfinite(fs) or fs <= 0:
        return {}
    x = x[:min(len(x), 200000)]
    f, p = welch(x, fs=fs, nperseg=min(2048, len(x)))
    bands = {"delta_0_4": (0.5, 4), "theta_4_8": (4, 8), "alpha_8_13": (8, 13),
             "beta_13_30": (13, 30), "gamma_30_45": (30, 45)}
    out = {}
    for name, (lo, hi) in bands.items():
        mask = (f >= lo) & (f < min(hi, fs / 2))
        out[name] = float(np.trapz(p[mask], f[mask])) if mask.sum() > 1 else None
    return out


def analyse_table(path: Path, ext: str) -> dict[str, Any]:
    import pandas as pd
    df = load_table(path, ext)
    if df.empty or len(df.columns) < 2:
        raise ValueError("The table does not contain usable sample rows and channel columns.")
    channels = finite_channel_columns(df)
    if not channels:
        raise ValueError("No numeric EEG channel columns were found.")
    time_col = next((c for c in df.columns if str(c).strip().lower() in {"time", "time_s", "timestamp"}), None)
    fs = None
    duration = None
    if time_col:
        tv = pd.to_numeric(df[time_col], errors="coerce").dropna().to_numpy(dtype=float)
        diffs = np.diff(tv)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if len(diffs):
            fs = float(1 / np.median(diffs))
            duration = float(tv[-1] - tv[0]) if len(tv) > 1 else 0.0
    channel_metrics = {}
    for c in channels[:128]:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        finite = v[np.isfinite(v)]
        if not len(finite):
            continue
        channel_metrics[str(c)] = {
            "samples": int(len(v)), "valid_samples": int(len(finite)),
            "missing_percent": round(100 * (1 - len(finite) / max(len(v), 1)), 2),
            "mean": float(np.mean(finite)), "std": float(np.std(finite)),
            "rms": float(np.sqrt(np.mean(finite ** 2))), "min": float(np.min(finite)),
            "max": float(np.max(finite)), "peak_to_peak": float(np.ptp(finite)),
            "band_power": spectral_bands(finite, fs) if fs and fs >= 32 else None
        }
    label_col = next((c for c in df.columns if str(c).strip().lower() in {"task_label", "label", "condition"}), None)
    trial_col = next((c for c in df.columns if str(c).strip().lower() == "trial_id"), None)
    labels = sorted(str(x) for x in df[label_col].dropna().unique()) if label_col else []
    trials = int(df[trial_col].nunique()) if trial_col else 0
    return {"kind": "tabular_eeg", "rows": int(len(df)), "columns": [str(c) for c in df.columns],
            "channel_count": len(channels), "channel_names": [str(c) for c in channels],
            "sampling_rate_estimate_hz": fs, "duration_seconds": duration,
            "label_column_present": bool(label_col), "trial_id_present": bool(trial_col),
            "label_classes": labels, "unique_trials": trials,
            "supervised_ml_candidate": bool(label_col and trial_col and len(labels) >= 2 and trials >= 6),
            "channel_metrics": channel_metrics,
            "limitations": ["Exploratory signal metrics only; units and acquisition metadata must be verified.",
                            "Frequency-band values are not a clinical diagnosis."]}


def analyse_edf(path: Path) -> dict[str, Any]:
    try:
        import mne
        raw = mne.io.read_raw(path, preload=False, verbose="ERROR")
        sf = float(raw.info["sfreq"])
        n = int(raw.n_times)
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        names = [raw.ch_names[i] for i in picks]
        summary = {"kind": "edf_bdf", "sampling_rate_hz": sf, "duration_seconds": n / sf,
                   "sample_count": n, "channel_count": len(raw.ch_names), "eeg_channel_count": len(names),
                   "channel_names": raw.ch_names, "eeg_channels": names,
                   "annotations": [{"onset": float(a["onset"]), "duration": float(a["duration"]), "description": str(a["description"])} for a in raw.annotations[:500]],
                   "limitations": ["This is file-structure and signal-metadata review, not clinical EEG interpretation.",
                                   "No seizure detector or validated clinical classifier is included."]}
        if len(picks) and n:
            stop = min(n, int(sf * 30))
            data = raw.get_data(picks=picks[:min(32, len(picks))], start=0, stop=stop)
            summary["first_30s_channel_rms"] = {raw.ch_names[picks[i]]: float(np.sqrt(np.mean(np.square(data[i])))) for i in range(data.shape[0])}
        raw.close()
        return summary
    except Exception as e:
        raise ValueError(f"EDF/BDF reader failed: {e}")


def analyse_pdf(path: Path, run_ocr: bool = True) -> tuple[dict[str, Any], str]:
    from server.report_wording import review_report_wording
    doc = pdf_document(path)
    from server.pdf_ocr import extract_scanned_page_text
    texts = []
    ocr_attempted = 0
    ocr_success = 0
    ocr_unavailable = False
    ocr_page_limit_reached = False
    for index, page in enumerate(doc):
        page_text = page.get_text("text") or ""
        if len(page_text.strip()) < 25 and run_ocr and index < 5:
            scanned_text, ocr_result = extract_scanned_page_text(page, index, enabled=True)
            if ocr_result == "tesseract_unavailable":
                ocr_unavailable = True
            elif ocr_result != "page_limit":
                ocr_attempted += 1
            if scanned_text:
                page_text = (page_text + "\n" + scanned_text).strip()
                ocr_success += 1
        elif len(page_text.strip()) < 25 and run_ocr and index >= 5:
            ocr_page_limit_reached = True
        texts.append(page_text)
    text = "\n\n".join(f"[PAGE {i+1}]\n{t}" for i, t in enumerate(texts) if t.strip())
    if not run_ocr:
        ocr_status = "disabled"
    elif ocr_unavailable:
        ocr_status = "unavailable"
    elif ocr_success and ocr_success == ocr_attempted:
        ocr_status = "completed"
    elif ocr_success:
        ocr_status = "partially_completed"
    elif ocr_attempted:
        ocr_status = "attempted_no_text"
    elif ocr_page_limit_reached:
        ocr_status = "page_limit"
    else:
        ocr_status = "not_needed"
    result = {"kind": "pdf_document", "page_count": len(doc),
              "pages_with_extracted_text": sum(bool(t.strip()) for t in texts),
              "text_extraction": "embedded PDF text and optional OCR" if text.strip() else "no readable text extracted",
              "extracted_characters": len(text),
              "ocr_status": ocr_status,
              "ocr_pages_attempted": ocr_attempted,
              "ocr_pages_with_text": ocr_success,
              "report_wording_review": review_report_wording(text),
              "limitations": ["Scanned page images are not interpreted as raw EEG waveforms.",
                              "A PDF report is not a substitute for the original digital EEG recording."]}
    doc.close()
    return result, text


@app.post("/api/files/{file_id}/analyse")
def analyse_file(file_id: str, ocr: bool = True, _: None = Depends(require_auth)):
    rec = get_file_record(file_id)
    ext, path = rec["extension"], rec["path"]
    try:
        extracted_text = ""
        if ext in {".csv", ".tsv", ".txt"}:
            summary = analyse_table(path, ext)
        elif ext in {".edf", ".bdf"}:
            summary = analyse_edf(path)
        elif ext == ".pdf":
            summary, extracted_text = analyse_pdf(path, run_ocr=ocr)
        elif ext in {".png", ".jpg", ".jpeg", ".webp"}:
            from PIL import Image
            im = Image.open(path)
            summary = {"kind": "image", "width": im.width, "height": im.height, "format": im.format,
                       "ocr_available": False,
                       "limitations": ["This is an image attachment, not calibrated raw EEG data.",
                                       "Visual review must be performed by a qualified reviewer."]}
        elif ext == ".json":
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            summary = {"kind": "json", "top_level_keys": list(payload)[:100] if isinstance(payload, dict) else [],
                       "json_type": type(payload).__name__, "limitations": ["Imported JSON fields are treated as supplied metadata."]}
            extracted_text = json.dumps(payload, ensure_ascii=False)[:50000]
        else:
            summary = {"kind": "text", "characters": path.stat().st_size}
            extracted_text = path.read_text(encoding="utf-8", errors="replace")[:50000]
        analysis_id = str(uuid.uuid4())
        with db() as con:
            con.execute("INSERT INTO analyses(id,file_id,study_id,created_at,kind,summary_json,extracted_text) VALUES(?,?,?,?,?,?,?)",
                        (analysis_id, file_id, rec["study_id"], now_iso(), summary.get("kind", "unknown"), json.dumps(summary), extracted_text))
            con.execute("UPDATE files SET status='analysed',error=NULL WHERE id=?", (file_id,))
        return {"id": analysis_id, "file_id": file_id, "summary": summary, "extracted_text": extracted_text[:12000]}
    except Exception as e:
        with db() as con:
            con.execute("UPDATE files SET status='error',error=? WHERE id=?", (str(e)[:1000], file_id))
        raise HTTPException(422, f"Analysis could not be completed: {e}")


@app.get("/api/analyses")
def list_analyses(study_id: str | None = None, _: None = Depends(require_auth)):
    with db() as con:
        if study_id:
            rows = con.execute("SELECT a.*,f.original_name FROM analyses a JOIN files f ON f.id=a.file_id WHERE a.study_id=? ORDER BY a.created_at DESC", (study_id,)).fetchall()
        else:
            rows = con.execute("SELECT a.*,f.original_name FROM analyses a JOIN files f ON f.id=a.file_id ORDER BY a.created_at DESC LIMIT 100").fetchall()
    out = []
    for r in rows:
        item = dict(r)
        item["summary"] = json.loads(item.pop("summary_json"))
        out.append(item)
    return out


def trial_feature_matrix(paths: list[Path]):
    import pandas as pd
    # Keep separate channel-schema cohorts so unrelated CSV exports cannot poison
    # a valid labelled dataset simply because they have different channel sets.
    cohorts: dict[tuple[str, ...], dict[str, Any]] = {}
    for path in paths:
        ext = path.suffix.lower()
        if ext not in {".csv", ".tsv", ".txt"}:
            continue
        try:
            df = load_table(path, ext)
        except Exception:
            continue
        lower = {str(c).strip().lower(): c for c in df.columns}
        lab = lower.get("task_label") or lower.get("label") or lower.get("condition")
        trial = lower.get("trial_id")
        if lab is None or trial is None:
            continue
        chans = finite_channel_columns(df)
        if not chans:
            continue
        names = tuple(str(c) for c in chans)
        cohort = cohorts.setdefault(names, {"X": [], "y": [], "groups": [], "sources": set()})
        for tid, g in df.groupby(trial, dropna=True):
            labs = [str(x) for x in g[lab].dropna().unique()]
            if len(labs) != 1:
                continue
            vals = []
            for c in chans:
                x = pd.to_numeric(g[c], errors="coerce").dropna().to_numpy(dtype=float)
                if len(x) < 8:
                    vals.extend([0.0, 0.0, 0.0, 0.0])
                else:
                    dx = np.diff(x)
                    vals.extend([float(np.mean(x)), float(np.std(x)), float(np.sqrt(np.mean(x*x))),
                                 float(np.mean(np.abs(dx))) if len(dx) else 0.0])
            cohort["X"].append(vals)
            cohort["y"].append(labs[0])
            cohort["groups"].append(f"{path.name}::{tid}")
            cohort["sources"].add(path.name)
    eligible = []
    diagnostics = []
    from collections import Counter
    for names, cohort in cohorts.items():
        counts = Counter(cohort["y"])
        diagnostics.append({"channels": len(names), "trials": len(cohort["y"]), "classes": dict(counts)})
        if len(counts) >= 2 and min(counts.values()) >= 3:
            eligible.append((len(cohort["y"]), names, cohort))
    if not eligible:
        raise ValueError("No compatible labelled dataset cohort has at least 2 classes and 3 independent trials per class. Check task_label/trial_id columns and use identical numeric channel columns. Cohorts found: " + json.dumps(diagnostics))
    _, names, cohort = max(eligible, key=lambda item: item[0])
    feature_names = [f"{c}__{m}" for c in names for m in ("mean", "std", "rms", "line_length")]
    return (np.asarray(cohort["X"], dtype=float), np.asarray(cohort["y"]),
            np.asarray(cohort["groups"]), feature_names, sorted(cohort["sources"]))


@app.post("/api/ml/train")
def train_model(_: None = Depends(require_auth)):
    with db() as con:
        files = con.execute("SELECT * FROM files WHERE extension IN ('.csv','.tsv','.txt') ORDER BY uploaded_at").fetchall()
    usable = [UPLOAD_DIR / r["stored_name"] for r in files if (UPLOAD_DIR / r["stored_name"]).exists()]
    if not usable:
        raise HTTPException(400, "Upload labelled research CSV/TSV data first.")
    try:
        X, y, groups, feature_names, training_files = trial_feature_matrix(usable)
        from collections import Counter
        counts = Counter(y.tolist())
        if len(counts) < 2:
            raise ValueError("At least two task-label classes are required.")
        if min(counts.values()) < 3:
            raise ValueError("Each class needs at least 3 independent labelled trials (minimum 2 for training and 1 held out for testing).")
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
        test_fraction = max(0.25, len(counts) / len(y))
        test_fraction = min(test_fraction, 0.4)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=42, stratify=y)
        clf = Pipeline([("scale", StandardScaler()), ("svm", SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True, random_state=42))])
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        labels = sorted(counts)
        metrics = {"algorithm": "StandardScaler + RBF-SVM baseline", "trials_total": int(len(y)),
                   "trials_train": int(len(y_train)), "trials_test": int(len(y_test)),
                   "class_counts": dict(counts), "accuracy": float(accuracy_score(y_test, pred)),
                   "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                   "chance_level_reference": float(1 / len(labels)),
                   "classification_report": classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0),
                   "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
                   "labels": labels,
                   "warning": "Exploratory within-dataset baseline only. A single random trial split is not independent validation; do not use for clinical decisions or unrestricted thought decoding."}
        model_id = str(uuid.uuid4())
        model_path = MODEL_DIR / f"{model_id}.joblib"
        import joblib
        joblib.dump(clf, model_path)
        rev = current_revision()
        with db() as con:
            con.execute("INSERT INTO models(id,created_at,model_path,metrics_json,feature_names_json,classes_json,dataset_revision,training_files_json) VALUES(?,?,?,?,?,?,?,?)",
                        (model_id, now_iso(), str(model_path), json.dumps(metrics), json.dumps(feature_names), json.dumps(labels), rev, json.dumps(training_files)))
        return {"id": model_id, "created_at": now_iso(), "metrics": metrics, "dataset_revision": rev, "feature_count": len(feature_names)}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Model training failed: {e}")


@app.get("/api/ml/models")
def list_models(_: None = Depends(require_auth)):
    rev = current_revision()
    with db() as con:
        rows = con.execute("SELECT * FROM models ORDER BY created_at DESC LIMIT 30").fetchall()
    result = []
    for r in rows:
        result.append({"id": r["id"], "created_at": r["created_at"], "metrics": json.loads(r["metrics_json"]),
                       "classes": json.loads(r["classes_json"]), "dataset_revision": r["dataset_revision"],
                       "stale": r["dataset_revision"] < rev})
    return {"dataset_revision": rev, "models": result}


def trial_prediction_matrix(path: Path, expected_feature_names: list[str]):
    import pandas as pd
    df = load_table(path, path.suffix.lower())
    lower = {str(c).strip().lower(): c for c in df.columns}
    trial = lower.get("trial_id")
    if trial is None:
        raise ValueError("Prediction CSV/TSV needs a trial_id column.")
    channels = finite_channel_columns(df)
    channel_names = [str(c) for c in channels]
    expected_channels = []
    for item in expected_feature_names:
        channel = item.rsplit("__", 1)[0]
        if channel not in expected_channels:
            expected_channels.append(channel)
    if channel_names != expected_channels:
        raise ValueError("Input channel columns do not match the trained model.")
    rows, trial_ids = [], []
    for tid, group in df.groupby(trial, dropna=True):
        values = []
        for c in channels:
            x = pd.to_numeric(group[c], errors="coerce").dropna().to_numpy(dtype=float)
            if len(x) < 8:
                values.extend([0.0, 0.0, 0.0, 0.0])
            else:
                dx = np.diff(x)
                values.extend([float(np.mean(x)), float(np.std(x)), float(np.sqrt(np.mean(x*x))),
                               float(np.mean(np.abs(dx))) if len(dx) else 0.0])
        rows.append(values)
        trial_ids.append(str(tid))
    if not rows:
        raise ValueError("No usable trials found in prediction file.")
    return np.asarray(rows, dtype=float), trial_ids


@app.post("/api/ml/predict/{model_id}/{file_id}")
def predict_file(model_id: str, file_id: str, _: None = Depends(require_auth)):
    rec = get_file_record(file_id)
    if rec["extension"] not in {".csv", ".tsv", ".txt"}:
        raise HTTPException(415, "Prediction input must be a CSV/TSV with trial_id and numeric channel columns.")
    with db() as con:
        model_row = con.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
    if not model_row:
        raise HTTPException(404, "Model not found")
    try:
        stored_features = json.loads(model_row["feature_names_json"])
        X, trial_ids = trial_prediction_matrix(rec["path"], stored_features)
        import joblib
        model = joblib.load(model_row["model_path"])
        preds = model.predict(X)
        probs = model.predict_proba(X)
        return {"model_id": model_id, "file_id": file_id,
                "predictions": [{"trial_id": trial_ids[i], "predicted_task_label": str(preds[i]),
                                 "probabilities": {str(c): float(probs[i][j]) for j, c in enumerate(model.classes_)}}
                                for i in range(len(preds))],
                "warning": "Research-only task classification. It is not a decoded private thought, diagnosis or clinical communication result."}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/studies/{study_id}/report.pdf")
def study_report(study_id: str, _: None = Depends(require_auth)):
    with db() as con:
        study = con.execute("SELECT * FROM studies WHERE id=?", (study_id,)).fetchone()
        if not study:
            raise HTTPException(404, "Study not found")
        files = con.execute("SELECT * FROM files WHERE study_id=? ORDER BY uploaded_at", (study_id,)).fetchall()
        analyses = con.execute("SELECT a.*,f.original_name FROM analyses a JOIN files f ON f.id=a.file_id WHERE a.study_id=? ORDER BY a.created_at", (study_id,)).fetchall()
        selected_pages = con.execute("SELECT p.file_id,p.page_num,f.original_name,f.stored_name,f.extension FROM study_pages p JOIN files f ON f.id=p.file_id WHERE p.study_id=? ORDER BY f.original_name,p.page_num", (study_id,)).fetchall()
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm)
        styles = getSampleStyleSheet()
        story = [Paragraph("NEUROBRIDGE — EEG RESEARCH WORKBENCH", styles["Title"]),
                 Paragraph("Technical review summary · Draft for qualified reviewer", styles["Heading2"]), Spacer(1, 8)]
        s = dict(study)
        details = [["Study title", s.get("title") or "—"], ["De-identified subject code", s.get("subject_code") or "—"],
                   ["Age / sex", f"{s.get('age') or '—'} / {s.get('sex') or '—'}"], ["Study date", s.get("study_date") or "—"],
                   ["Montage", s.get("montage") or "—"], ["Sampling rate", str(s.get("sampling_rate") or "—")]]
        t = Table(details, colWidths=[48*mm, 125*mm])
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),colors.HexColor("#eaf4f5")),("GRID",(0,0),(-1,-1),0.3,colors.grey),
                               ("VALIGN",(0,0),(-1,-1),"TOP"),("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),("PADDING",(0,0),(-1,-1),6)]))
        story += [t, Spacer(1, 10), Paragraph("Uploaded files", styles["Heading2"])]
        file_rows = [["File", "Type", "Size (MB)", "Status"]]
        for f in files:
            file_rows.append([f["original_name"], f["extension"], f"{f['size_bytes']/1048576:.2f}", f["status"]])
        if len(file_rows) == 1:
            file_rows.append(["No files uploaded", "—", "—", "—"])
        ft = Table(file_rows, repeatRows=1, colWidths=[90*mm, 18*mm, 25*mm, 40*mm])
        ft.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dceef1")),("GRID",(0,0),(-1,-1),0.3,colors.grey),
                                ("PADDING",(0,0),(-1,-1),5),("VALIGN",(0,0),(-1,-1),"TOP")]))
        story += [ft, Spacer(1, 10), Paragraph("Technical analyses", styles["Heading2"])]
        if not analyses:
            if files:
                empty_message = (
                    "Files are attached to this study, but no analysis has been saved yet. "
                    "Open Signal Review, choose each PDF, run Analyse, and wait for the "
                    "saved confirmation. Then generate this report again."
                )
            else:
                empty_message = (
                    "No files are attached to this study, so no PDF text or EEG report wording "
                    "could be reviewed. Open the Uploads section and upload the EEG report while "
                    "this study is selected. If the file was uploaded earlier, use Attach unassigned "
                    "files. Then open Signal Review, run Analyse on the PDF, and generate this report again."
                )
            story.append(Paragraph(empty_message, styles["BodyText"]))
        for a in analyses:
            summ = json.loads(a["summary_json"])
            story.append(Paragraph(a["original_name"], styles["Heading3"]))
            lines = []
            for k in ["kind", "rows", "sample_count", "duration_seconds", "sampling_rate_estimate_hz", "channel_count", "page_count", "pages_with_embedded_text", "pages_with_extracted_text", "ocr_status", "ocr_pages_attempted", "ocr_pages_with_text"]:
                if k in summ:
                    lines.append(f"<b>{k.replace('_',' ').title()}:</b> {summ[k]}")
            story.append(Paragraph("<br/>".join(lines) or "No summary fields available.", styles["BodyText"]))
            wording_review = summ.get("report_wording_review")
            if wording_review:
                from xml.sax.saxutils import escape as xml_escape
                story.append(Paragraph("Automated report-wording review", styles["Heading4"]))
                for finding in wording_review.get("items", []):
                    status_label = str(finding.get("status", "context_uncertain")).replace("_", " ")
                    story.append(Paragraph(
                        f"<b>{xml_escape(str(finding.get('finding', 'finding')))} — {xml_escape(status_label)}</b>",
                        styles["BodyText"]
                    ))
                    for evidence in finding.get("evidence", [])[:3]:
                        evidence_text = f"Page {evidence.get('page', '?')}: {evidence.get('wording', '')}"
                        story.append(Paragraph(xml_escape(evidence_text), styles["BodyText"]))
                story.append(Paragraph(xml_escape(str(wording_review.get("note", ""))), styles["Italic"]))
            story.append(Spacer(1, 6))
        if s.get("source_text"):
            story += [Paragraph("Source report text supplied", styles["Heading2"]), Paragraph(str(s["source_text"])[:6000].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>"), styles["BodyText"])]
        story += [Spacer(1, 12), Paragraph("IMPORTANT CLINICAL SAFETY & DATA LIMITATIONS", styles["Heading2"]),
                  Paragraph("This document is an automated technical summary generated from uploaded files and user-supplied study metadata. It is NOT a clinical EEG report, diagnosis, seizure-exclusion statement, treatment recommendation, or validated medical-device output. The software does not read unrestricted thoughts or establish clinical abnormalities. Original complete EEG, acquisition settings, clinical history and artifacts must be reviewed by a qualified neurologist/neurophysiologist. Incomplete screenshots or selected pages cannot exclude events outside the supplied material.", styles["BodyText"]),
                  Spacer(1, 8), Paragraph("Generated by NeuroBridge v3.0 · " + now_iso(), styles["Italic"])]
        if selected_pages:
            import fitz
            story += [PageBreak(), Paragraph("Selected EEG report pages for reviewer context", styles["Heading1"]),
                      Paragraph("These are original PDF page images selected by the user. They have not been clinically interpreted by this application.", styles["BodyText"]), Spacer(1, 8)]
            for item in selected_pages:
                max_w, max_h = 170*mm, 235*mm
                if item["extension"] == ".pdf":
                    pdoc = fitz.open(UPLOAD_DIR / item["stored_name"])
                    page = pdoc[int(item["page_num"]) - 1]
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                    image_data = io.BytesIO(pix.tobytes("png"))
                    width_px, height_px = pix.width, pix.height
                    pdoc.close()
                    caption = f"{item['original_name']} — page {item['page_num']}"
                else:
                    from PIL import Image as PILImage
                    im = PILImage.open(UPLOAD_DIR / item["stored_name"]).convert("RGB")
                    width_px, height_px = im.size
                    image_data = io.BytesIO()
                    im.save(image_data, format="PNG")
                    image_data.seek(0)
                    caption = f"{item['original_name']} — uploaded waveform image"
                scale = min(max_w / width_px, max_h / height_px)
                image_data.seek(0)
                story += [Paragraph(caption, styles["Heading2"]),
                          RLImage(image_data, width=width_px*scale, height=height_px*scale), Spacer(1, 8)]
        doc.build(story)
        buf.seek(0)
        return StreamingResponse(buf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="NeuroBridge_{study_id[:8]}_technical_draft.pdf"'})
    except Exception as e:
        raise HTTPException(500, f"Report generation failed: {e}")


@app.get("/", response_class=HTMLResponse)
def root():
    return (FRONTEND / "index.html").read_text(encoding="utf-8")

app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
