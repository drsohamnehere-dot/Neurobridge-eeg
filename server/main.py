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

app = FastAPI(title="NeuroBridge EEG Workbench API", version="3.1.0")

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
    if not row:
        return None
    result = dict(row)
    # Normalise legacy records too: earlier uploads may have stored uppercase extensions.
    if "extension" in result:
        fallback = Path(result.get("original_name") or "").suffix
        result["extension"] = str(result.get("extension") or fallback).lower()
    return result


def bump_dataset_revision() -> int:
    with db() as con:
        con.execute("UPDATE app_meta SET value=CAST(value AS INTEGER)+1 WHERE key='dataset_revision'")
        return int(con.execute("SELECT value FROM app_meta WHERE key='dataset_revision'").fetchone()[0])


def current_revision() -> int:
    with db() as con:
        return int(con.execute("SELECT value FROM app_meta WHERE key='dataset_revision'").fetchone()[0])


@app.get("/api/health")
def health():
    return {"ok": True, "service": "NeuroBridge API", "version": "3.1.0", "storage": "persistent volume required for cloud hosting"}


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
        attached = cur.rowcount
        # Analyses created before a file was attached must follow that file into the study.
        con.execute(
            """UPDATE analyses SET study_id=?
               WHERE study_id IS NULL
               AND file_id IN (SELECT id FROM files WHERE study_id=?)""",
            (study_id, study_id),
        )
        return {"attached": attached}


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


def analyse_edf(path: Path, channel_limit: int = 64) -> dict[str, Any]:
    raw = None
    try:
        import mne
        from server.eeg_signal_review import (
            SUPPORTED_CHANNEL_LIMITS,
            apply_channel_limit,
            build_signal_review,
            scalp_eeg_channel_indices,
        )

        if channel_limit not in SUPPORTED_CHANNEL_LIMITS:
            raise ValueError(f"Channel limit must be one of {SUPPORTED_CHANNEL_LIMITS}.")
        raw = mne.io.read_raw(path, preload=False, verbose="ERROR")
        sf = float(raw.info["sfreq"])
        n = int(raw.n_times)
        if not np.isfinite(sf) or sf <= 0 or n <= 0:
            raise ValueError("The recording has invalid sampling metadata or no samples.")
        picks = scalp_eeg_channel_indices(raw.ch_names)
        selection_note = "Scalp EEG channels identified from standard 10-20/10-10 electrode labels."
        if not picks:
            picks = [i for i, kind in enumerate(raw.get_channel_types()) if kind == "eeg"]
            selection_note = "No standard scalp electrode labels were recognised; EDF EEG channel types used as fallback. Verify channel selection manually."
        names = [raw.ch_names[i] for i in picks]
        duration = n / sf
        selected_set = set(picks)
        auxiliary_names = [name for i, name in enumerate(raw.ch_names) if i not in selected_set]
        summary = {
            "kind": "edf_bdf",
            "sampling_rate_hz": sf,
            "duration_seconds": duration,
            "sample_count": n,
            "channel_count": len(raw.ch_names),
            "eeg_channel_count": len(names),
            "available_scalp_channel_count": len(picks),
            "requested_channel_capacity": channel_limit,
            "auxiliary_channel_count": len(auxiliary_names),
            "channel_names": raw.ch_names,
            "eeg_channels": names,
            "auxiliary_channels": auxiliary_names,
            "channel_selection_note": selection_note,
            "annotations": [
                {"onset": float(a["onset"]), "duration": float(a["duration"]), "description": str(a["description"])}
                for a in raw.annotations[:500]
            ],
            "limitations": [
                "Quantitative features currently cover the initial 120-second analysis window, not the complete recording.",
                "Transient candidates are heuristic review prompts, not epileptiform-discharge or seizure detections.",
                "No validated clinical EEG classifier is included; expert review of the raw EEG is required.",
            ],
        }
        if picks and n:
            selected_picks = apply_channel_limit(picks, channel_limit)
            window_seconds = min(120.0, duration)
            stop = min(n, max(1, int(round(sf * window_seconds))))
            data = raw.get_data(picks=selected_picks, start=0, stop=stop, reject_by_annotation=None)
            selected_names = [raw.ch_names[i] for i in selected_picks]
            summary["signal_review"] = build_signal_review(
                data_volts=data,
                sfreq=sf,
                channel_names=selected_names,
                recording_duration_seconds=duration,
                window_start_seconds=0.0,
            )
            summary["analysis_channels_requested"] = channel_limit
            summary["analysis_channels_processed"] = len(selected_picks)
            summary["analysis_channel_limit_note"] = (
                f"Requested capacity: {channel_limit}; actual scalp EEG channels analysed: {len(selected_picks)}."
                if len(selected_picks) < channel_limit
                else f"Processed {len(selected_picks)} scalp EEG channels."
            )
        else:
            summary["signal_review"] = {
                "status": "not_available",
                "interpretation": "No scalp EEG channels were identified in this file.",
                "channels_processed": 0,
            }
        return summary
    except Exception as e:
        raise ValueError(f"EDF/BDF reader failed: {e}")
    finally:
        if raw is not None:
            raw.close()



@app.post("/api/files/{file_id}/find-clean-segments")
def find_clean_eeg_segments(
    file_id: str,
    epoch_seconds: int = 2,
    channel_limit: int = 24,
    target_state: str = "any",
    spectrum_min_hz: float = 0.5,
    spectrum_max_hz: float = 30.0,
    delta_low_hz: float = 0.5,
    delta_high_hz: float = 4.0,
    theta_low_hz: float = 4.0,
    theta_high_hz: float = 8.0,
    alpha_low_hz: float = 8.0,
    alpha_high_hz: float = 13.0,
    beta_lf_low_hz: float = 13.0,
    beta_lf_high_hz: float = 20.0,
    beta_hf_low_hz: float = 20.0,
    beta_hf_high_hz: float = 30.0,
    window_function: str = "none",
    _: None = Depends(require_auth),
):
    """Find ranked low-artifact candidate epochs across an EDF/BDF recording.

    This is conservative heuristic screening, not proof of artifact-free EEG.
    """
    if epoch_seconds not in {2, 4}:
        raise HTTPException(422, "epoch_seconds must be 2 or 4.")
    if channel_limit not in {24, 32, 64}:
        raise HTTPException(422, "channel_limit must be 24, 32 or 64.")
    if target_state not in {"any", "eyes_closed", "eyes_open", "alert_awake", "drowsy_or_sleep"}:
        raise HTTPException(422, "Unsupported target_state.")

    rec = get_file_record(file_id)
    if rec["extension"] not in {".edf", ".bdf"}:
        raise HTTPException(415, "Low-artifact segment screening requires an EDF/BDF recording.")

    raw = None
    try:
        import re
        import mne
        from server.eeg_signal_review import (
            assess_eeg_epoch_artifacts,
            apply_channel_limit,
            aggregate_fft_band_power,
            calculate_fft_rhythm_indices,
            merge_candidate_epoch_intervals,
            rhythm_ratio_rows_from_power_sums,
            scalp_eeg_channel_indices,
        )

        raw = mne.io.read_raw(rec["path"], preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        sample_count = int(raw.n_times)
        duration = sample_count / sfreq if sfreq > 0 else 0.0
        if not np.isfinite(sfreq) or sfreq <= 0 or sample_count <= 0:
            raise HTTPException(422, "The EDF/BDF header has invalid sampling metadata.")

        scalp_picks = scalp_eeg_channel_indices(raw.ch_names)
        selection_note = "Scalp channels identified from standard electrode labels."
        if not scalp_picks:
            scalp_picks = [i for i, kind in enumerate(raw.get_channel_types()) if kind == "eeg"]
            selection_note = "Standard scalp labels were not recognised; MNE EEG channel types were used. Verify this channel selection."
        selected_picks = apply_channel_limit(scalp_picks, channel_limit)
        if not selected_picks:
            raise HTTPException(422, "No scalp EEG channels were identified. Check the EDF channel labels or add a channel map.")
        channel_types = raw.get_channel_types()
        auxiliary_channels = [
            {"name": raw.ch_names[index], "type": channel_types[index]}
            for index in range(len(raw.ch_names)) if index not in set(scalp_picks)
        ]
        eog_channels = [
            name for name, kind in zip(raw.ch_names, channel_types)
            if kind == "eog" or "EOG" in name.upper()
        ]
        ecg_channels = [
            name for name, kind in zip(raw.ch_names, channel_types)
            if kind == "ecg" or "ECG" in name.upper()
        ]
        emg_channels = [
            name for name, kind in zip(raw.ch_names, channel_types)
            if kind == "emg" or "EMG" in name.upper()
        ]

        annotations = [
            {
                "onset": float(item["onset"]),
                "duration": max(0.0, float(item["duration"])),
                "description": str(item["description"] or ""),
            }
            for item in raw.annotations
        ]

        def state_from_description(description: str) -> str | None:
            text = re.sub(r"[_-]+", " ", description.casefold()).strip()
            compact = re.sub(r"\s+", " ", text)
            words = set(compact.split())
            if "eyes closed" in compact or "close eyes" in compact or "closed eyes" in compact or compact in {"ec", "ec marker"}:
                return "eyes_closed"
            if "eyes open" in compact or "open eyes" in compact or compact in {"eo", "eo marker"}:
                return "eyes_open"
            if "drowsy" in words or "sleep" in words or "asleep" in words:
                return "drowsy_or_sleep"
            if "awake" in words or "alert" in words or compact in {"wake", "w"}:
                return "alert_awake"
            return None

        state_events = sorted(
            [
                {"onset": item["onset"], "state": state_from_description(item["description"]), "description": item["description"]}
                for item in annotations if state_from_description(item["description"]) is not None
            ],
            key=lambda item: item["onset"],
        )
        if target_state != "any" and not any(item["state"] == target_state for item in state_events):
            label = {"eyes_closed": "eyes-closed", "eyes_open": "eyes-open", "alert_awake": "alert/awake", "drowsy_or_sleep": "drowsy/sleep"}[target_state]
            raise HTTPException(
                422,
                f"No {label} state markers were recognised in this EDF/BDF file. Use 'All states — verify manually' or add clear event annotations before filtering by state.",
            )

        bad_annotations = [
            item for item in annotations
            if item["description"].strip().casefold().startswith(("bad", "artifact", "artefact"))
        ]
        epoch_samples = int(round(epoch_seconds * sfreq))
        if epoch_samples < 32:
            raise HTTPException(422, "The selected epoch is too short for this sampling rate.")
        epochs_per_chunk = max(1, int((30.0 * sfreq) // epoch_samples))
        chunk_samples = epochs_per_chunk * epoch_samples

        spectral_settings = {
            "spectrum_min_hz": spectrum_min_hz,
            "spectrum_max_hz": spectrum_max_hz,
            "delta_low_hz": delta_low_hz,
            "delta_high_hz": delta_high_hz,
            "theta_low_hz": theta_low_hz,
            "theta_high_hz": theta_high_hz,
            "alpha_low_hz": alpha_low_hz,
            "alpha_high_hz": alpha_high_hz,
            "beta_lf_low_hz": beta_lf_low_hz,
            "beta_lf_high_hz": beta_lf_high_hz,
            "beta_hf_low_hz": beta_hf_low_hz,
            "beta_hf_high_hz": beta_hf_high_hz,
            "window_function": window_function,
        }
        # Validate the user-selected frequency configuration before reading the full file.
        calculate_fft_rhythm_indices(
            data_volts=np.zeros((1, max(32, epoch_seconds * int(round(sfreq)) + 1))),
            sfreq=sfreq,
            channel_names=["validation"],
            settings=spectral_settings,
        )
        epoch_results: list[dict[str, Any]] = []
        band_rows: list[tuple[float, dict[str, float]]] = []
        spectral_power_sums_by_state: dict[str, dict[str, dict[str, float]]] = {}
        accepted_epoch_counts_by_state: dict[str, int] = {}
        skipped_bad = 0
        skipped_transition = 0
        skipped_state = 0
        epochs_scanned = 0

        def active_state_at(moment: float) -> dict[str, Any] | None:
            active = None
            for event in state_events:
                if event["onset"] <= moment:
                    active = event
                else:
                    break
            return active

        for chunk_start in range(0, sample_count, chunk_samples):
            chunk_stop = min(sample_count, chunk_start + chunk_samples)
            chunk_data = raw.get_data(
                picks=selected_picks,
                start=chunk_start,
                stop=chunk_stop,
                reject_by_annotation=None,
            )
            local_start = 0
            while local_start + epoch_samples <= chunk_data.shape[1]:
                absolute_start_sample = chunk_start + local_start
                start_seconds = absolute_start_sample / sfreq
                end_seconds = (absolute_start_sample + epoch_samples) / sfreq
                midpoint = (start_seconds + end_seconds) / 2.0

                active_state = active_state_at(midpoint)
                state_name = active_state["state"] if active_state else "unlabelled"
                if target_state != "any" and state_name != target_state:
                    skipped_state += 1
                    local_start += epoch_samples
                    continue
                if active_state and start_seconds < active_state["onset"] + 3.0:
                    skipped_transition += 1
                    local_start += epoch_samples
                    continue
                if any(
                    (item["duration"] > 0 and item["onset"] < end_seconds and item["onset"] + item["duration"] > start_seconds)
                    or (item["duration"] == 0 and start_seconds <= item["onset"] < end_seconds)
                    for item in bad_annotations
                ):
                    skipped_bad += 1
                    local_start += epoch_samples
                    continue

                epoch_data = chunk_data[:, local_start:local_start + epoch_samples]
                assessment = assess_eeg_epoch_artifacts(
                    data_volts=epoch_data,
                    sfreq=sfreq,
                    channel_names=[raw.ch_names[i] for i in selected_picks],
                )
                epochs_scanned += 1
                epoch_results.append({
                    "start_seconds": round(start_seconds, 3),
                    "end_seconds": round(end_seconds, 3),
                    "duration_seconds": round(epoch_seconds, 3),
                    "state": state_name,
                    "status": assessment["status"],
                    "quality_score": assessment["quality_score"],
                    "channels_with_hard_flags": assessment["channels_with_hard_flags"],
                    "channels_with_review_flags": assessment["channels_with_review_flags"],
                    "flagged_channels": [
                        {"channel": item["channel"], "flags": item.get("flags", [])}
                        for item in assessment["channels"] if item.get("flags")
                    ],
                    "median_channel_band_power_pct": assessment["median_channel_band_power_pct"],
                })
                if assessment["status"] == "low_artifact_candidate":
                    band_rows.append((midpoint, assessment["median_channel_band_power_pct"]))
                    fft_result = calculate_fft_rhythm_indices(
                        data_volts=epoch_data,
                        sfreq=sfreq,
                        channel_names=[raw.ch_names[i] for i in selected_picks],
                        settings=spectral_settings,
                    )
                    state_bucket = spectral_power_sums_by_state.setdefault(state_name, {})
                    aggregate_fft_band_power(state_bucket, fft_result)
                    accepted_epoch_counts_by_state[state_name] = accepted_epoch_counts_by_state.get(state_name, 0) + 1
                local_start += epoch_samples

        if not epoch_results:
            raise HTTPException(422, "No eligible EEG epochs were available after state, transition and bad-annotation filtering.")

        intervals = merge_candidate_epoch_intervals(epoch_results)
        low_artifact_epochs = sum(1 for item in epoch_results if item["status"] == "low_artifact_candidate")
        low_artifact_seconds = round(low_artifact_epochs * epoch_seconds, 1)
        longest_run = max((item["duration_seconds"] for item in intervals), default=0.0)
        clean_seconds_by_state: dict[str, float] = {}
        for item in epoch_results:
            if item["status"] == "low_artifact_candidate":
                state_key = str(item.get("state") or "unlabelled")
                clean_seconds_by_state[state_key] = clean_seconds_by_state.get(state_key, 0.0) + epoch_seconds
        clean_seconds_by_state = {key: round(value, 1) for key, value in clean_seconds_by_state.items()}
        identified_state_keys = {"eyes_closed", "eyes_open", "alert_awake", "drowsy_or_sleep"}
        identified_state_seconds = {
            key: value for key, value in clean_seconds_by_state.items()
            if key in identified_state_keys
        }
        longest_state_seconds = max(identified_state_seconds.values(), default=0.0)
        state_specific_60s_met = longest_state_seconds >= 60.0 and (
            target_state != "any" or bool(state_events)
        )
        flag_counts: dict[str, int] = {}
        for item in epoch_results:
            for channel in item["flagged_channels"]:
                for flag in channel["flags"]:
                    flag_counts[flag] = flag_counts.get(flag, 0) + 1

        split_half = None
        if len(band_rows) >= 4:
            midpoint_index = len(band_rows) // 2
            first_half = band_rows[:midpoint_index]
            second_half = band_rows[midpoint_index:]
            band_comparison = {}
            for band in ("delta", "theta", "alpha", "beta", "low_gamma"):
                first_values = [row[1][band] for row in first_half if band in row[1]]
                second_values = [row[1][band] for row in second_half if band in row[1]]
                if first_values and second_values:
                    first_median = float(np.median(first_values))
                    second_median = float(np.median(second_values))
                    band_comparison[band] = {
                        "first_half_median_pct": round(first_median, 2),
                        "second_half_median_pct": round(second_median, 2),
                        "absolute_difference_percentage_points": round(abs(first_median - second_median), 2),
                    }
            split_half = {
                "method": "Chronological first-half versus second-half comparison of accepted epochs.",
                "bands": band_comparison,
                "note": "Descriptive stability check only; differences do not prove or exclude artefact or pathology.",
            }

        best_epochs = sorted(
            [item for item in epoch_results if item["status"] == "low_artifact_candidate"],
            key=lambda item: (-item["quality_score"], item["start_seconds"]),
        )[:60]
        if not best_epochs:
            best_epochs = sorted(
                epoch_results,
                key=lambda item: (item["channels_with_hard_flags"], item["channels_with_review_flags"], -item["quality_score"], item["start_seconds"]),
            )[:20]

        spectral_ratio_analysis_by_state = {}
        for state_name, power_sums in spectral_power_sums_by_state.items():
            spectral_ratio_analysis_by_state[state_name] = {
                "method": "FFT bin-power sum over accepted low-artifact candidate epochs",
                "window_function": window_function,
                "settings": spectral_settings,
                "accepted_epochs": accepted_epoch_counts_by_state.get(state_name, 0),
                "channels": rhythm_ratio_rows_from_power_sums(power_sums, spectral_settings),
                "formulae": {
                    "alpha_theta": "Alpha index / Theta index",
                    "theta_beta": "Theta index / (Beta LF index + Beta HF index)",
                    "theta_beta_lf": "Theta index / Beta LF index",
                },
                "note": "Indices are calculated only from low-artifact candidate epochs in this state. This is heuristic segment selection, not proof of artifact-free EEG.",
            }

        state_labeling = (
            "State selected from recognised EDF/BDF event annotations."
            if target_state != "any" else
            ("State markers were found; all states are included and must not be mixed for state-specific qEEG."
             if state_events else
             "No recognised eyes-open/eyes-closed/awake markers were found. State is unlabelled; manually select one condition and do not mix states.")
        )
        summary = {
            "kind": "clean_segment_screening",
            "method": "Full-recording heuristic low-artifact candidate epoch screening",
            "file_id": file_id,
            "filename": rec["original_name"],
            "recording_duration_seconds": round(duration, 3),
            "sampling_rate_hz": sfreq,
            "available_scalp_channel_count": len(scalp_picks),
            "channels_assessed": len(selected_picks),
            "requested_channel_capacity": channel_limit,
            "auxiliary_channel_count": len(auxiliary_channels),
            "auxiliary_channels": auxiliary_channels,
            "eog_channels_available": eog_channels,
            "ecg_channels_available": ecg_channels,
            "emg_channels_available": emg_channels,
            "channel_selection_note": selection_note,
            "epoch_seconds": epoch_seconds,
            "epoch_overlap_seconds": 0,
            "spectral_settings": spectral_settings,
            "spectral_ratio_analysis_by_state": spectral_ratio_analysis_by_state,
            "accepted_low_artifact_epochs_by_state": accepted_epoch_counts_by_state,
            "target_state": target_state,
            "state_labeling_note": state_labeling,
            "transition_guard_seconds": 3,
            "epochs_scanned": epochs_scanned,
            "epochs_skipped_bad_annotations": skipped_bad,
            "epochs_skipped_state_mismatch": skipped_state,
            "epochs_skipped_state_transition": skipped_transition,
            "low_artifact_epochs": low_artifact_epochs,
            "low_artifact_seconds_total": low_artifact_seconds,
            "minimum_60_seconds_available": low_artifact_seconds >= 60.0,
            "low_artifact_seconds_by_state": clean_seconds_by_state,
            "minimum_60_seconds_in_one_identified_state": state_specific_60s_met,
            "longest_candidate_state_seconds": round(longest_state_seconds, 1),
            "longest_contiguous_candidate_seconds": round(longest_run, 3),
            "candidate_intervals": intervals[:100],
            "best_epochs": best_epochs,
            "review_required_epochs": epochs_scanned - low_artifact_epochs,
            "artifact_flag_counts": flag_counts,
            "split_half_consistency": split_half,
            "thresholds_note": (
                "Thresholds are conservative engineering heuristics for prioritising visual review, not clinically validated artefact detectors. "
                "An epoch with no flags is only a low-artifact candidate; subtle eye, ECG, muscle, sweat, drowsiness and electrode artefacts may remain. "
                f"EOG channels available: {len(eog_channels)}; ECG channels available: {len(ecg_channels)}; EMG channels available: {len(emg_channels)}. "
                "These auxiliary channels are listed but are not yet used for automatic regression or ICA artefact correction."
            ),
            "clinical_note": (
                "Do not interpret this screen as proof that an epoch is artefact-free or physiologically normal. "
                "A qualified EEG reviewer must inspect candidate segments in the original recording. "
                "Potential epileptiform events must not be discarded solely because of amplitude or transient heuristics. Drowsiness is not independently classified unless it is identified in recording annotations."
            ),
            "state_consistency_note": (
                "The 60-second target is only considered met within one identified state when clear state annotations exist or a state filter was selected. "
                "When state is unlabelled, or several states are mixed, total candidate seconds must not be treated as a single-state qEEG sample."
            ),
        }
        analysis_id = str(uuid.uuid4())
        with db() as con:
            con.execute(
                "INSERT INTO analyses(id,file_id,study_id,created_at,kind,summary_json,extracted_text) VALUES(?,?,?,?,?,?,?)",
                (analysis_id, file_id, rec.get("study_id"), now_iso(), "clean_segment_screening", json.dumps(summary), ""),
            )
        return {"id": analysis_id, "file_id": file_id, "summary": summary}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"Low-artifact segment scan failed: {e}")
    finally:
        if raw is not None:
            raw.close()


@app.get("/api/files/{file_id}/edf-preview")
def edf_waveform_preview(
    file_id: str,
    start_seconds: float = 0.0,
    duration_seconds: float = 10.0,
    max_channels: int = 24,
    _: None = Depends(require_auth),
):
    """Return a compact, downsampled waveform preview from a stored EDF/BDF."""
    rec = get_file_record(file_id)
    if rec["extension"] not in {".edf", ".bdf"}:
        raise HTTPException(415, "Waveform preview is available only for EDF/BDF recordings.")
    raw = None
    try:
        import mne
        from scipy.signal import resample
        from server.eeg_signal_review import SUPPORTED_CHANNEL_LIMITS, scalp_eeg_channel_indices

        if max_channels < 1 or max_channels > 64:
            raise HTTPException(422, "max_channels must be between 1 and 64.")
        raw = mne.io.read_raw(rec["path"], preload=False, verbose="ERROR")
        sf = float(raw.info["sfreq"])
        total_seconds = float(raw.n_times / sf)
        if not np.isfinite(sf) or sf <= 0 or raw.n_times <= 0:
            raise ValueError("The recording has invalid sampling metadata.")
        picks = scalp_eeg_channel_indices(raw.ch_names)
        if not picks:
            picks = [i for i, kind in enumerate(raw.get_channel_types()) if kind == "eeg"]
        if not picks:
            raise HTTPException(422, "No EEG channels could be identified in this EDF/BDF file.")
        duration = min(15.0, max(2.0, float(duration_seconds)))
        start = min(max(0.0, float(start_seconds)), max(0.0, total_seconds - duration))
        stop = min(raw.n_times, max(1, int(round((start + duration) * sf))))
        first = min(raw.n_times - 1, int(round(start * sf)))
        selected = picks[:max_channels]
        names = [raw.ch_names[i] for i in selected]
        data = raw.get_data(picks=selected, start=first, stop=stop, reject_by_annotation=None) * 1_000_000.0
        target_sfreq = min(64.0, sf)
        target_count = max(2, int(round(data.shape[1] * target_sfreq / sf)))
        preview = resample(data, target_count, axis=1)
        times = start + (np.arange(preview.shape[1], dtype=float) / target_sfreq)
        return {
            "kind": "edf_waveform_preview",
            "start_seconds": round(start, 3),
            "duration_seconds": round(float(preview.shape[1] / target_sfreq), 3),
            "recording_duration_seconds": round(total_seconds, 3),
            "source_sampling_rate_hz": sf,
            "preview_sampling_rate_hz": target_sfreq,
            "requested_channel_capacity": max_channels,
            "available_scalp_channel_count": len(picks),
            "channels_displayed": len(selected),
            "channels": [
                {"name": name, "values_uv": np.round(preview[index], 2).tolist()}
                for index, name in enumerate(names)
            ],
            "time_seconds": np.round(times, 3).tolist(),
            "note": "Downsampled visual preview only; use the original EDF/BDF for detailed review.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"Unable to create EDF/BDF waveform preview: {e}")
    finally:
        if raw is not None:
            raw.close()


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
def analyse_file(file_id: str, ocr: bool = True, channel_limit: int = 64, _: None = Depends(require_auth)):
    rec = get_file_record(file_id)
    ext, path = rec["extension"], rec["path"]
    try:
        extracted_text = ""
        if ext in {".csv", ".tsv", ".txt"}:
            summary = analyse_table(path, ext)
        elif ext in {".edf", ".bdf"}:
            from server.eeg_signal_review import SUPPORTED_CHANNEL_LIMITS
            if channel_limit not in SUPPORTED_CHANNEL_LIMITS:
                raise HTTPException(
                    422,
                    f"channel_limit must be one of {SUPPORTED_CHANNEL_LIMITS}.",
                )
            summary = analyse_edf(path, channel_limit=channel_limit)
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
    # A report request also prepares any attached digital EEG recordings that have not
    # yet been analysed. This is descriptive research screening, not clinical diagnosis.
    with db() as con:
        study = con.execute("SELECT * FROM studies WHERE id=?", (study_id,)).fetchone()
        if not study:
            raise HTTPException(404, "Study not found")
        file_rows = con.execute("SELECT * FROM files WHERE study_id=? ORDER BY uploaded_at", (study_id,)).fetchall()

    for file_row in file_rows:
        rec = dict(file_row)
        extension = str(rec.get("extension") or Path(rec.get("original_name") or "").suffix).lower()
        if extension not in {".edf", ".bdf"}:
            continue
        recording_path = UPLOAD_DIR / rec["stored_name"]
        if not recording_path.exists():
            with db() as con:
                con.execute(
                    "UPDATE files SET status='error',error=? WHERE id=?",
                    ("Original recording is not present in the current server storage. Re-upload the EDF/BDF file.", rec["id"]),
                )
            continue
        with db() as con:
            existing = con.execute(
                "SELECT 1 FROM analyses WHERE file_id=? AND kind='edf_bdf' LIMIT 1",
                (rec["id"],),
            ).fetchone()
        if existing:
            continue
        try:
            summary = analyse_edf(recording_path)
            with db() as con:
                con.execute(
                    """INSERT INTO analyses(id,file_id,study_id,created_at,kind,summary_json,extracted_text)
                       VALUES(?,?,?,?,?,?,?)""",
                    (str(uuid.uuid4()), rec["id"], study_id, now_iso(), "edf_bdf", json.dumps(summary), ""),
                )
                con.execute("UPDATE files SET status='analysed',error=NULL WHERE id=?", (rec["id"],))
        except Exception as e:
            with db() as con:
                con.execute(
                    "UPDATE files SET status='error',error=? WHERE id=?",
                    ("Automatic EDF/BDF screening failed: " + str(e)[:400], rec["id"]),
                )

    with db() as con:
        study = con.execute("SELECT * FROM studies WHERE id=?", (study_id,)).fetchone()
        files = con.execute("SELECT * FROM files WHERE study_id=? ORDER BY uploaded_at", (study_id,)).fetchall()
        # Join through the file's current study assignment so older analyses are not lost
        # when their source file was attached to a study after the analysis was run.
        analyses = con.execute(
            """SELECT a.*,f.original_name,f.extension,f.size_bytes,f.status
               FROM analyses a JOIN files f ON f.id=a.file_id
               WHERE f.study_id=? ORDER BY a.created_at""",
            (study_id,),
        ).fetchall()
        selected_pages = con.execute("SELECT p.file_id,p.page_num,f.original_name,f.stored_name,f.extension FROM study_pages p JOIN files f ON f.id=p.file_id WHERE p.study_id=? ORDER BY f.original_name,p.page_num", (study_id,)).fetchall()
    if not files:
        raise HTTPException(
            400,
            "No files are attached to this study. Select the correct study, tick 'Attach unassigned uploads' on the report page, or upload the original EDF/BDF recording first.",
        )
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
                 Paragraph("Comprehensive EEG technical analysis · Clinician review draft", styles["Heading2"]),
                 Paragraph("Automated descriptive screening only — clinical interpretation must be completed by a qualified EEG specialist.", styles["Italic"]), Spacer(1, 8)]
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
            file_rows.append(["No files attached to this study", "—", "—", "—"])
        else:
            for row_index, file_row in enumerate(files, start=1):
                stored_path = UPLOAD_DIR / file_row["stored_name"]
                if not stored_path.exists():
                    file_rows[row_index][3] = "Missing from server — re-upload required"
        ft = Table(file_rows, repeatRows=1, colWidths=[90*mm, 18*mm, 25*mm, 40*mm])
        ft.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dceef1")),("GRID",(0,0),(-1,-1),0.3,colors.grey),
                                ("PADDING",(0,0),(-1,-1),5),("VALIGN",(0,0),(-1,-1),"TOP")]))
        story += [ft, Spacer(1, 10), Paragraph("Technical analyses", styles["Heading2"])]
        if not analyses:
            if files:
                empty_message = (
                    "Files are attached to this study, but no readable saved analysis is available. "
                    "Check the file status above. If a raw EDF/BDF is listed as missing, re-upload it; "
                    "if the format was rejected, inspect the server error before relying on this report."
                )
            else:
                empty_message = (
                    "No files are attached to this study. Before generating a useful EEG report, "
                    "select the correct study and attach the intended upload from the Uploads section. "
                    "The report action can then run the available bounded EDF/BDF technical screening."
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
            clean_screen = summ if summ.get("kind") == "clean_segment_screening" else None
            if clean_screen:
                from xml.sax.saxutils import escape as xml_escape
                story.append(Paragraph("Low-artifact segment screening — heuristic review only", styles["Heading4"]))
                clean_overview = [
                    f"Epoch length: {clean_screen.get('epoch_seconds', '—')} s; epochs scanned: {clean_screen.get('epochs_scanned', 0)}",
                    f"Scalp channels assessed: {clean_screen.get('channels_assessed', 0)} of {clean_screen.get('available_scalp_channel_count', 0)} available",
                    f"Low-artifact candidate epochs: {clean_screen.get('low_artifact_epochs', 0)}; candidate seconds total: {clean_screen.get('low_artifact_seconds_total', 0)}",
                    f"Longest continuous candidate interval: {clean_screen.get('longest_contiguous_candidate_seconds', 0)} s",
                    f"At least 60 seconds total: {'Yes' if clean_screen.get('minimum_60_seconds_available') else 'No'}",
                    f"At least 60 seconds within one identified state: {'Yes' if clean_screen.get('minimum_60_seconds_in_one_identified_state') else 'Not verified'}",
                    f"Candidate seconds by state: {clean_screen.get('low_artifact_seconds_by_state', {})}",
                    f"Accepted candidate epochs by state: {clean_screen.get('accepted_low_artifact_epochs_by_state', {})}",
                    f"Skipped BAD/artifact annotation epochs: {clean_screen.get('epochs_skipped_bad_annotations', 0)}",
                    f"Skipped transition epochs: {clean_screen.get('epochs_skipped_state_transition', 0)}",
                ]
                story.append(Paragraph("<br/>".join(xml_escape(line) for line in clean_overview), styles["BodyText"]))
                clean_rows = [["Start (s)", "End (s)", "Duration (s)", "Epochs"]]
                for interval in clean_screen.get("candidate_intervals", [])[:40]:
                    clean_rows.append([
                        str(interval.get("start_seconds", "")),
                        str(interval.get("end_seconds", "")),
                        str(interval.get("duration_seconds", "")),
                        str(interval.get("epoch_count", "")),
                    ])
                if len(clean_rows) > 1:
                    clean_table = Table(clean_rows, repeatRows=1, colWidths=[30*mm, 30*mm, 35*mm, 25*mm])
                    clean_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dceef1")),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                        ("PADDING", (0, 0), (-1, -1), 4),
                        ("FONTSIZE", (0, 0), (-1, -1), 7),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]))
                    story.append(clean_table)
                story.append(Paragraph(xml_escape(clean_screen.get("thresholds_note", "")), styles["BodyText"]))
                story.append(Paragraph(xml_escape(clean_screen.get("clinical_note", "")), styles["BodyText"]))
                story.append(Paragraph(xml_escape(clean_screen.get("state_consistency_note", "")), styles["BodyText"]))
            def append_spectral_ratio_table(title, ratio_analysis):
                if not ratio_analysis:
                    return
                from xml.sax.saxutils import escape as xml_escape
                settings = ratio_analysis.get("settings") or {}
                range_hz = ratio_analysis.get("effective_spectrum_range_hz")
                if not range_hz:
                    range_hz = [settings.get("spectrum_min_hz", 0.5), settings.get("spectrum_max_hz", 30.0)]
                story.append(Paragraph(title, styles["Heading4"]))
                story.append(Paragraph(
                    f"FFT-power indices · total range {range_hz[0]}–{range_hz[1]} Hz · "
                    f"window: {ratio_analysis.get('window_function', settings.get('window_function', '—'))} · "
                    f"Beta LF: {settings.get('beta_lf_low_hz', '—')}–{settings.get('beta_lf_high_hz', '—')} Hz · "
                    f"Beta HF: {settings.get('beta_hf_low_hz', '—')}–{settings.get('beta_hf_high_hz', '—')} Hz",
                    styles["BodyText"],
                ))
                ratio_rows = [[
                    "Channel", "Delta %", "Theta %", "Alpha %",
                    "Beta LF %", "Beta HF %", "Alpha/Theta",
                    "Theta/Beta", "Theta/Beta LF",
                ]]
                for item in ratio_analysis.get("channels", [])[:64]:
                    indices = item.get("band_power_index_pct") or {}
                    ratios = item.get("ratios") or {}
                    fmt = lambda value: "—" if value is None else f"{float(value):.3f}"
                    ratio_rows.append([
                        str(item.get("channel", "")),
                        fmt(indices.get("delta")),
                        fmt(indices.get("theta")),
                        fmt(indices.get("alpha")),
                        fmt(indices.get("beta_lf")),
                        fmt(indices.get("beta_hf")),
                        fmt(ratios.get("alpha_theta")),
                        fmt(ratios.get("theta_beta")),
                        fmt(ratios.get("theta_beta_lf")),
                    ])
                if len(ratio_rows) > 1:
                    ratio_table = Table(
                        ratio_rows, repeatRows=1,
                        colWidths=[17*mm, 17*mm, 17*mm, 17*mm, 19*mm, 19*mm, 22*mm, 21*mm, 22*mm],
                    )
                    ratio_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dceef1")),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                        ("PADDING", (0, 0), (-1, -1), 3),
                        ("FONTSIZE", (0, 0), (-1, -1), 6.5),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]))
                    story.append(ratio_table)
                story.append(Paragraph(
                    "Definitions: Alpha/Theta = alpha index ÷ theta index; "
                    "Theta/Beta = theta index ÷ (beta LF index + beta HF index); "
                    "Theta/Beta LF = theta index ÷ beta LF index. Rhythm indices use summed FFT-bin power in each band divided by total FFT-bin power in the configured spectrum range × 100%.",
                    styles["BodyText"],
                ))
                if ratio_analysis.get("note"):
                    story.append(Paragraph(xml_escape(str(ratio_analysis["note"])), styles["BodyText"]))

            regular_ratio_analysis = summ.get("spectral_ratio_analysis")
            if regular_ratio_analysis:
                append_spectral_ratio_table("FFT rhythm indices and spectral ratios (all analysed EDF window)", regular_ratio_analysis)
            clean_ratio_by_state = summ.get("spectral_ratio_analysis_by_state") or {}
            for state_name, ratio_analysis in clean_ratio_by_state.items():
                if ratio_analysis.get("channels"):
                    append_spectral_ratio_table(
                        "Clean-segment FFT rhythm indices and ratios — " + str(state_name).replace("_", " "),
                        ratio_analysis,
                    )

            signal_review = summ.get("signal_review")
            if signal_review:
                from xml.sax.saxutils import escape as xml_escape
                story.append(Paragraph("Quantitative EEG signal screening", styles["Heading4"]))
                overview = [
                    f"Analysis window: {signal_review.get('analysis_window_start_seconds', 0)}–"
                    f"{signal_review.get('analysis_window_start_seconds', 0) + signal_review.get('analysis_window_seconds', 0)} s",
                    f"Channels processed: {signal_review.get('channels_processed', 0)}",
                    f"Usable channels: {signal_review.get('usable_channels', 0)}",
                    f"Channels with quality flags: {signal_review.get('channels_with_quality_flags', 0)}",
                    f"Generic transient candidates: {signal_review.get('transient_candidate_count', 0)}",
                ]
                story.append(Paragraph("<br/>".join(xml_escape(x) for x in overview), styles["BodyText"]))
                if s.get("sampling_rate") and summ.get("sampling_rate_hz"):
                    entered_rate = float(s["sampling_rate"])
                    header_rate = float(summ["sampling_rate_hz"])
                    if abs(entered_rate - header_rate) > max(0.05, header_rate * 0.005):
                        story.append(Paragraph(
                            f"<b>Acquisition metadata check:</b> study form says {entered_rate:g} Hz, "
                            f"but the EDF/BDF header reports {header_rate:g} Hz. Use the original device/export settings to resolve this discrepancy.",
                            styles["BodyText"],
                        ))
                channel_rows = [["Channel", "RMS (µV)", "Peak-to-peak (µV)", "Dominant Hz", "Quality flags"]]
                for channel in signal_review.get("channels", [])[:64]:
                    if not channel.get("usable"):
                        channel_rows.append([str(channel.get("channel", "")), "—", "—", "—", str(channel.get("reason", "unavailable"))[:60]])
                        continue
                    channel_rows.append([
                        str(channel.get("channel", "")),
                        str(channel.get("rms_uV", "—")),
                        str(channel.get("peak_to_peak_uV", "—")),
                        str(channel.get("dominant_frequency_hz") or "—"),
                        ", ".join(channel.get("quality_flags", []))[:70] or "None flagged",
                    ])
                channel_table = Table(channel_rows, repeatRows=1, colWidths=[20*mm, 24*mm, 30*mm, 22*mm, 70*mm])
                channel_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dceef1")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("PADDING", (0, 0), (-1, -1), 4),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]))
                story.append(channel_table)
                candidates = signal_review.get("transient_candidates_top25", [])
                if candidates:
                    story.append(Paragraph("Generic transient candidates for manual review (not spike/seizure detections)", styles["Heading5"]))
                    candidate_lines = [
                        f"{item.get('channel')} at {item.get('time_seconds')} s · {item.get('amplitude_uV')} µV"
                        for item in candidates[:15]
                    ]
                    story.append(Paragraph("<br/>".join(xml_escape(x) for x in candidate_lines), styles["BodyText"]))
                story.append(Paragraph(xml_escape(str(signal_review.get("interpretation", ""))), styles["Italic"]))
                story.append(Paragraph(xml_escape(str(signal_review.get("coverage_note", ""))), styles["BodyText"]))
                story.append(Paragraph(xml_escape(str(signal_review.get("thresholds_note", ""))), styles["BodyText"]))

                # Add per-channel relative band-power values to make the screening evidence inspectable.
                band_definitions = [
                    ("Delta 0.5–4 Hz", "delta_0_4_hz"),
                    ("Theta 4–8 Hz", "theta_4_8_hz"),
                    ("Alpha 8–13 Hz", "alpha_8_13_hz"),
                    ("Beta 13–30 Hz", "beta_13_30_hz"),
                    ("Low gamma 30–45 Hz", "low_gamma_30_45_hz"),
                ]
                band_rows = [["Channel"] + [item[0] for item in band_definitions]]
                for channel in signal_review.get("channels", [])[:64]:
                    if not channel.get("usable"):
                        continue
                    relative = channel.get("relative_band_power_pct") or {}
                    band_rows.append(
                        [str(channel.get("channel", ""))]
                        + [("—" if relative.get(key) is None else f"{float(relative[key]):.1f}%")
                           for _, key in band_definitions]
                    )
                if len(band_rows) > 1:
                    story.append(Paragraph("Relative spectral power by channel (descriptive)", styles["Heading4"]))
                    band_table = Table(band_rows, repeatRows=1,
                                       colWidths=[22*mm, 30*mm, 30*mm, 30*mm, 30*mm, 30*mm])
                    band_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dceef1")),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                        ("PADDING", (0, 0), (-1, -1), 3),
                        ("FONTSIZE", (0, 0), (-1, -1), 6.5),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]))
                    story.append(band_table)

                # The application does not score these clinical EEG criteria; they remain for specialist review.
                story.append(Paragraph("Clinical EEG review checklist — not automatically assessed", styles["Heading4"]))
                review_items = [
                    "Background organization and posterior dominant rhythm",
                    "Background frequency, amplitude, continuity and reactivity",
                    "Interhemispheric symmetry and synchrony",
                    "Focal or diffuse slowing and its clinical correlation",
                    "Sleep architecture and normal sleep transients, if recorded",
                    "Epileptiform discharges and electrographic seizures",
                    "Activation procedures (hyperventilation / photic stimulation), if performed",
                    "Movement, muscle, electrode and ECG artefacts",
                    "Event markers and clinical video / symptom correlation",
                ]
                story.append(Paragraph("<br/>".join("- " + xml_escape(item) for item in review_items), styles["BodyText"]))
                story.append(Paragraph(
                    "<b>Automated impression:</b> No validated normal/abnormal EEG classification, "
                    "epileptiform-discharge diagnosis, or seizure diagnosis is generated by this software. "
                    "The descriptive findings above require review of the complete original recording by a qualified specialist.",
                    styles["BodyText"],
                ))
                annotations = summ.get("annotations") or []
                story.append(Paragraph(
                    f"EDF/BDF annotation markers available in file metadata: {len(annotations)}. "
                    "Annotation markers are acquisition metadata and are not, by themselves, verified clinical events.",
                    styles["BodyText"],
                ))
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
                  Paragraph("This is a comprehensive technical research summary and clinician-review worksheet, NOT a final clinical EEG interpretation, diagnosis, seizure-exclusion statement, treatment recommendation, or validated medical-device output. Automated quantitative processing is currently limited to the initial bounded analysis window (up to 120 seconds) and descriptive signal-quality / band-power features. Generic transient candidates are not epileptiform discharges. The application does not determine background normality, posterior dominant rhythm, focal dysfunction, epileptiform activity, or seizures. A qualified neurologist/neurophysiologist must review the complete original recording, montage, filters, sensitivity, state, activation procedures, artifacts and clinical history before documenting a clinical impression.", styles["BodyText"]),
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
