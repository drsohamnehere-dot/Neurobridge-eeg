#!/usr/bin/env python3
"""NeuroBridge one-file PDF research screening app for Render.

Keeps the original private login, upload, SQLite metadata, and download workflow.
Adds PDF text indexing, rendered-page visual descriptors, unsupervised page-level
outlier prioritisation, and a downloadable clinician-review report. The ML component
is not a clinically trained EEG classifier and does not diagnose or exclude disease.
Use anonymised research/test material and follow institutional privacy rules.
"""

import os
import re
import json
import uuid
import sqlite3
import mimetypes
import logging
import fitz
import io
import math
import numpy as np
from PIL import Image
from sklearn.ensemble import IsolationForest
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from xml.sax.saxutils import escape
from pathlib import Path
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote
from email.parser import BytesParser
from email import policy

APP_NAME = "NEUROBRIDGE"
TARGET_CHANNEL_CONFIGURATION = 24  # Project target only; actual montage must be read from each recording.
MAX_ANALYSIS_PAGES = 120
PORT = int(os.environ.get("PORT", "10000"))
ACCESS_CODE = os.environ.get("NB_ACCESS_CODE", "")
DATA_DIR = Path(os.environ.get("NB_DATA_DIR", "/tmp/neurobridge_data"))
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "neurobridge.sqlite3"
ANALYSIS_DIR = DATA_DIR / "analysis_reports"
MAX_FILE_BYTES = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".tsv", ".txt", ".edf", ".bdf", ".pdf", ".png", ".jpg", ".jpeg"}
TOKENS = set()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEUROBRIDGE EEG Workbench</title>
<style>
:root{color-scheme:dark;--bg:#07111e;--panel:#102238;--line:#29445e;--muted:#9fb0c3;--cyan:#65dce2;--white:#edf5ff;--green:#69d5a5;--amber:#e7bd6b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--white);font:16px system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:22px max(18px,calc((100% - 1100px)/2));border-bottom:1px solid var(--line);background:#091827}header h1{margin:0;font-size:23px;letter-spacing:2px}header p{margin:5px 0 0;color:var(--muted);font-size:12px;letter-spacing:1px}
main{max-width:1100px;margin:24px auto;padding:0 16px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;margin-bottom:18px}.notice{background:#302817;border:1px solid #69542b;color:#f4e5bd;border-radius:12px;padding:13px;font-size:13px;line-height:1.5;margin-bottom:18px}
label{display:block;font-size:13px;color:#c2d1e1;margin:12px 0 6px}input{width:100%;padding:13px;border-radius:10px;border:1px solid var(--line);background:#081522;color:white;font:inherit}button{background:var(--cyan);color:#06202a;border:0;border-radius:10px;padding:10px 14px;font-weight:700;font-size:14px;cursor:pointer}button:disabled{opacity:.5;cursor:not-allowed}.secondary{background:#20364c;color:var(--white);border:1px solid var(--line)}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.drop{border:2px dashed #41627d;border-radius:14px;padding:28px 16px;text-align:center;background:#0a1a2a;margin:15px 0}.drop.drag{border-color:var(--cyan);background:#102b3a}.muted{color:var(--muted);font-size:13px}.hidden{display:none!important}.status{margin-top:12px;color:var(--cyan);white-space:pre-wrap;font-size:14px}.error{color:#ffb0aa}.progress{height:13px;border-radius:20px;background:#071522;overflow:hidden;margin:10px 0;border:1px solid var(--line)}.bar{height:100%;width:0;background:var(--cyan);transition:width .12s}.file{border-top:1px solid var(--line);padding:12px 0;display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap}.pill{font-size:11px;padding:4px 8px;border:1px solid var(--line);border-radius:30px;color:#c7d8e8}.small{font-size:12px;color:var(--muted)}
.results{margin-top:16px}.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}.metric{background:#0a1a2a;border:1px solid var(--line);border-radius:12px;padding:12px}.metric b{display:block;font-size:22px;color:var(--cyan)}.metric span{font-size:12px;color:var(--muted)}.result-table{width:100%;border-collapse:collapse;font-size:13px;margin-top:10px}.result-table th,.result-table td{border-bottom:1px solid var(--line);text-align:left;padding:9px 7px;vertical-align:top}.result-table th{color:#c7d8e8}.flag{color:var(--amber);font-weight:700}.ok{color:var(--green);font-weight:700}.result-note{padding:12px;background:#0a1a2a;border:1px solid var(--line);border-radius:10px;line-height:1.5;font-size:13px;white-space:pre-wrap}.tag{display:inline-block;border:1px solid var(--line);padding:3px 7px;border-radius:20px;margin:3px;font-size:12px}
@media(max-width:600px){header{padding:18px}main{margin:15px auto}.panel{padding:15px}.result-table{font-size:12px}}
</style></head><body>
<header><h1>🧠 NEUROBRIDGE</h1><p>EEG PDF RESEARCH SCREENING · VISUAL OUTLIER REVIEW · CLINICIAN WORKFLOW</p></header>
<main>
<div class="notice"><b>Research screening, not a diagnostic device.</b> The local ML step prioritises pages that look visually different within the same PDF. It does not identify epilepsy, seizures, spikes, or a normal EEG. Source-text terms are displayed as text found in the report, not independently confirmed findings. A qualified EEG reader must inspect the original recording and approve any interpretation. Use anonymised files and follow your institution's data rules.</div>
<section class="panel" id="loginPanel"><h2>Private application access</h2><p class="muted">Enter the access code set by the owner in the Render environment settings.</p><label for="code">Access code</label><input id="code" type="password" autocomplete="current-password" placeholder="Your private access code"><div class="row" style="margin-top:14px"><button id="loginBtn" onclick="login()">Open application</button></div><div id="loginStatus" class="status"></div></section>
<section class="panel hidden" id="appPanel"><div class="row" style="justify-content:space-between"><div><h2>Upload EEG files</h2><p class="muted">PDF waveform reports are the primary input. CSV, TSV, TXT, EDF, BDF and images can also be stored. Maximum 50 MB per file. Project target: 24 channels; each report's actual montage is assessed separately.</p></div><button class="secondary" onclick="logout()">Sign out</button></div>
<div class="drop" id="drop"><div style="font-size:34px">⇧</div><b>Choose a file or drop it here</b><p class="muted">Upload percentage and transferred size appear below.</p><input id="fileInput" type="file" accept=".csv,.tsv,.txt,.edf,.bdf,.pdf,.png,.jpg,.jpeg"></div>
<div id="uploadArea" class="hidden"><div class="row" style="justify-content:space-between"><b id="uploadName">Uploading…</b><b id="pct">0%</b></div><div class="progress"><div class="bar" id="bar"></div></div><div class="small" id="uploadDetail">Waiting to upload</div></div><div id="uploadStatus" class="status"></div>
<div class="row" style="margin-top:18px;justify-content:space-between"><h3 style="margin:0">Files stored by this running server</h3><button class="secondary" onclick="loadFiles()">Refresh list</button></div><div id="files" class="muted" style="margin-top:10px">Loading…</div>
</section>
<section class="panel hidden results" id="analysisPanel"><div class="row" style="justify-content:space-between"><h2 style="margin:0">PDF screening results</h2><a id="reportLink" class="tag" href="#" target="_blank">Download review PDF</a></div><div id="analysisSummary" class="result-note" style="margin-top:12px"></div><div class="metric-grid" id="metricGrid"></div><h3>Page-by-page review</h3><div style="overflow-x:auto"><table class="result-table"><thead><tr><th>Page</th><th>Page classification</th><th>Visual outlier</th><th>Ink %</th><th>Visual density peaks</th></tr></thead><tbody id="pageRows"></tbody></table></div><h3>Terms found in extracted report text</h3><div id="termMatches" class="result-note">No terms processed.</div><h3>Screening interpretation</h3><div id="clinicalDraft" class="result-note"></div></section>
<section class="panel hidden" id="healthPanel"><b>Server status</b><div id="healthText" class="muted">Connected</div></section>
</main>
<script>
let token='';const $=id=>document.getElementById(id);
function status(id,msg,err=false){$(id).textContent=msg;$(id).className='status'+(err?' error':'')}
async function login(){const code=$('code').value;if(!code){status('loginStatus','Enter your access code.',true);return}$('loginBtn').disabled=true;try{const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({access_code:code})});const d=await r.json();if(!r.ok)throw Error(d.detail||'Login failed');token=d.token;$('loginPanel').classList.add('hidden');$('appPanel').classList.remove('hidden');$('healthPanel').classList.remove('hidden');status('healthText','Connected to NeuroBridge backend. Local PDF screening is available.');loadFiles()}catch(e){status('loginStatus',e.message,true)}finally{$('loginBtn').disabled=false}}
function logout(){token='';$('appPanel').classList.add('hidden');$('healthPanel').classList.add('hidden');$('analysisPanel').classList.add('hidden');$('loginPanel').classList.remove('hidden');$('code').value=''}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function loadFiles(){try{const r=await fetch('/api/files',{headers:{'X-NB-Token':token}});const d=await r.json();if(!r.ok)throw Error(d.detail||'Could not load files');if(!d.files.length){$('files').textContent='No uploads yet.';return}$('files').innerHTML=d.files.map(f=>`<div class="file"><div><b>${esc(f.name)}</b><div class="small">${(f.size/1048576).toFixed(2)} MB · ${esc(f.uploaded_at)}</div></div><span class="pill">${esc(f.extension.toUpperCase())}</span><div class="row"><a style="color:#65dce2" href="/api/download/${encodeURIComponent(f.id)}?token=${encodeURIComponent(token)}">Download</a>${f.extension.toLowerCase()==='.pdf'?`<button onclick="analyzePdf('${encodeURIComponent(f.id)}')">Run PDF screening</button>`:''}</div></div>`).join('')}catch(e){$('files').textContent=e.message}}
function upload(file){if(!file)return;if(file.size>50*1024*1024){status('uploadStatus','This file is larger than 50 MB. Please choose a smaller file.',true);return}const ext='.'+file.name.split('.').pop().toLowerCase();if(!['.csv','.tsv','.txt','.edf','.bdf','.pdf','.png','.jpg','.jpeg'].includes(ext)){status('uploadStatus','This file type is not supported.',true);return}$('uploadArea').classList.remove('hidden');$('uploadName').textContent=file.name;$('pct').textContent='0%';$('bar').style.width='0%';$('uploadDetail').textContent='Starting upload…';status('uploadStatus','');const fd=new FormData();fd.append('file',file);const x=new XMLHttpRequest();x.open('POST','/api/upload');x.setRequestHeader('X-NB-Token',token);x.upload.onprogress=e=>{if(e.lengthComputable){const p=Math.round(e.loaded/e.total*100);$('pct').textContent=p+'%';$('bar').style.width=p+'%';$('uploadDetail').textContent=(e.loaded/1048576).toFixed(2)+' MB of '+(e.total/1048576).toFixed(2)+' MB uploaded'}};x.onload=()=>{try{const d=JSON.parse(x.responseText);if(x.status<200||x.status>=300)throw Error(d.detail||'Upload failed');$('pct').textContent='100%';$('bar').style.width='100%';$('uploadDetail').textContent='Upload complete · saved on current server';status('uploadStatus','File uploaded successfully. Choose “Run PDF screening” beside the PDF.');loadFiles()}catch(e){status('uploadStatus',e.message,true)}};x.onerror=()=>status('uploadStatus','Network error. The upload did not finish.',true);x.send(fd)}
async function analyzePdf(id){$('analysisPanel').classList.add('hidden');status('uploadStatus','Screening PDF pages… this may take a little while for long reports.');try{const r=await fetch('/api/analyze/'+encodeURIComponent(id),{headers:{'X-NB-Token':token}});const d=await r.json();if(!r.ok)throw Error(d.detail||'PDF screening failed');$('analysisPanel').classList.remove('hidden');$('analysisSummary').textContent=d.summary+'\n\nML method: '+d.ml_method+'\nChannel configuration: '+d.channel_configuration+'\nPages with extracted text: '+d.pages_with_text+' of '+d.page_count+'\nReport: source text is not treated as a confirmed EEG finding.';$('metricGrid').innerHTML=[['PDF pages',d.page_count],['Waveform-like pages',d.waveform_like_pages],['Visual outlier pages',d.visual_outlier_pages],['Source-text term groups',Object.keys(d.source_text_terms||{}).length]].map(m=>`<div class="metric"><b>${esc(m[1])}</b><span>${esc(m[0])}</span></div>`).join('');$('pageRows').innerHTML=d.pages.map(p=>`<tr><td>${p.page_number}</td><td>${esc(p.page_class)}</td><td>${p.visual_outlier?'<span class="flag">Review</span>':'<span class="ok">Not flagged</span>'}</td><td>${(p.dark_pixel_fraction*100).toFixed(2)}%</td><td>${p.visual_density_peaks.length?esc(p.visual_density_peaks.join(', ')+'%'):'—'}</td></tr>`).join('');const terms=d.source_text_terms||{};const termHtml=Object.keys(terms).length?Object.entries(terms).map(([k,v])=>`<div><b>${esc(k)}:</b> ${v.map(x=>`<span class="tag">${esc(x.term)} · p.${x.pages.join(', ')}</span>`).join('')}</div>`).join(''):'No selected EEG terms were found in the selectable PDF text. Scanned page images may not have a text layer.';$('termMatches').innerHTML=termHtml;$('clinicalDraft').textContent=d.clinician_review_note;$('reportLink').href='/api/analysis-report/'+encodeURIComponent(d.analysis_id)+'?token='+encodeURIComponent(token);$('analysisPanel').scrollIntoView({behavior:'smooth',block:'start'});status('uploadStatus','PDF screening finished. Review the flagged pages and the original recording with an EEG specialist.')}catch(e){status('uploadStatus',e.message,true)}}
$('fileInput').addEventListener('change',e=>upload(e.target.files[0]));const drop=$('drop');drop.addEventListener('dragover',e=>{e.preventDefault();drop.classList.add('drag')});drop.addEventListener('dragleave',()=>drop.classList.remove('drag'));drop.addEventListener('drop',e=>{e.preventDefault();drop.classList.remove('drag');upload(e.dataTransfer.files[0])});
</script></body></html>'''


def json_bytes(data):
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def db_connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db_connect() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS uploads (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, stored_name TEXT NOT NULL,
            size INTEGER NOT NULL, mime TEXT NOT NULL, extension TEXT NOT NULL,
            uploaded_at TEXT NOT NULL
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY, file_id TEXT NOT NULL, created_at TEXT NOT NULL,
            result_json TEXT NOT NULL, report_name TEXT NOT NULL
        )""")


def safe_name(name):
    name = Path(name.replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    return (name or "upload.bin")[:120]
    
def render_eeg_pdf(pdf_path, zoom=1.5, max_pages=MAX_ANALYSIS_PAGES):
    """Render each PDF page as a PNG byte string for local visual screening."""
    rendered = []
    with fitz.open(pdf_path) as pdf:
        if pdf.is_encrypted:
            raise ValueError("PDF is password protected")
        if len(pdf) == 0:
            raise ValueError("PDF contains no pages")
        if len(pdf) > max_pages:
            raise ValueError(f"PDF has {len(pdf)} pages; the current limit is {max_pages}")
        for page_number, page in enumerate(pdf, start=1):
            rect = page.rect
            scale = min(float(zoom), 1700.0 / max(float(rect.width), float(rect.height), 1.0))
            scale = max(0.65, scale)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            rendered.append({
                "page_number": page_number,
                "image": pix.tobytes("png"),
                "width": pix.width,
                "height": pix.height,
            })
    return rendered


def extract_eeg_pdf_text(pdf_path, max_pages=MAX_ANALYSIS_PAGES):
    """Extract selectable text page by page. Image-only pages return empty text."""
    extracted = []
    with fitz.open(pdf_path) as pdf:
        if pdf.is_encrypted:
            raise ValueError("PDF is password protected")
        if len(pdf) == 0:
            raise ValueError("PDF contains no pages")
        if len(pdf) > max_pages:
            raise ValueError(f"PDF has {len(pdf)} pages; the current limit is {max_pages}")
        for page_number, page in enumerate(pdf, start=1):
            extracted.append({"page_number": page_number, "text": page.get_text("text").strip()})
    return extracted


def _page_visual_features(png_bytes):
    """Compute reproducible image descriptors; these are not EEG physiological features."""
    with Image.open(io.BytesIO(png_bytes)) as source:
        image = source.convert("L")
        image.thumbnail((1500, 1500))
        arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("Could not decode rendered PDF page")

    h, w = arr.shape
    y0, y1 = int(h * 0.04), max(int(h * 0.96), 1)
    x0, x1 = int(w * 0.025), max(int(w * 0.985), 1)
    crop = arr[y0:y1, x0:x1]
    # Dark-ink projection, robust to common white/grey paper backgrounds.
    ink = crop < 150
    total = max(int(ink.sum()), 1)
    dark_fraction = float(ink.mean())
    row_profile = ink.mean(axis=1)
    col_profile = ink.mean(axis=0)
    active_row_fraction = float(np.mean(row_profile > 0.0025))
    active_column_fraction = float(np.mean(col_profile > 0.0025))
    horizontal_adjacency = float(np.logical_and(ink[:, 1:], ink[:, :-1]).sum() / total) if ink.shape[1] > 1 else 0.0
    vertical_adjacency = float(np.logical_and(ink[1:, :], ink[:-1, :]).sum() / total) if ink.shape[0] > 1 else 0.0
    column_variation = float(np.std(col_profile))
    contrast = float(np.std(crop.astype(np.float32)) / 127.5)

    # Mark columns whose ink density is unusually high within this page.
    # This can highlight trace excursions, text, labels, or grid intersections.
    smooth = np.convolve(col_profile, np.ones(7, dtype=float) / 7.0, mode="same")
    median = float(np.median(smooth))
    mad = float(np.median(np.abs(smooth - median)))
    threshold = median + max(5.0 * 1.4826 * mad, 0.018)
    high = smooth > threshold
    groups = []
    start = None
    for idx, value in enumerate(high):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            if idx - start >= 2:
                groups.append((start, idx - 1))
            start = None
    if start is not None and len(high) - start >= 2:
        groups.append((start, len(high) - 1))
    peaks = []
    for a, b in groups:
        center = int((a + b) / 2)
        peaks.append(round(100.0 * center / max(len(col_profile) - 1, 1), 1))
    peaks = peaks[:12]

    return {
        "width": int(w),
        "height": int(h),
        "dark_pixel_fraction": round(dark_fraction, 6),
        "active_row_fraction": round(active_row_fraction, 5),
        "active_column_fraction": round(active_column_fraction, 5),
        "horizontal_adjacency": round(horizontal_adjacency, 5),
        "vertical_adjacency": round(vertical_adjacency, 5),
        "column_density_variation": round(column_variation, 5),
        "contrast_index": round(contrast, 5),
        "visual_density_peaks": peaks,
    }


EEG_TEXT_TERMS = {
    "Epileptiform wording": [
        ("epileptiform", r"\bepileptiform\b"),
        ("spike(s)", r"\bspikes?\b"),
        ("sharp wave(s)", r"\bsharp\s+waves?\b"),
        ("spike-and-wave", r"spike.?and.?wave"),
    ],
    "Seizure/ictal wording": [
        ("seizure", r"\bseizure\b"),
        ("ictal", r"\bictal\b"),
        ("electrographic event", r"\belectrographic event\b"),
    ],
    "Slowing wording": [
        ("focal slowing", r"\bfocal slowing\b"),
        ("generalized slowing", r"\bgeneralized slowing\b"),
        ("generalised slowing", r"\bgeneralised slowing\b"),
        ("diffuse slowing", r"\bdiffuse slowing\b"),
        ("slow wave", r"\bslow waves?\b"),
    ],
    "Background/rhythm wording": [
        ("posterior dominant rhythm", r"posterior dominant rhythm"),
        ("background rhythm", r"background rhythm"),
        ("alpha rhythm", r"alpha rhythm"),
        ("theta", r"\btheta\b"),
        ("delta", r"\bdelta\b"),
        ("beta", r"\bbeta\b"),
    ],
    "Activation/artifact wording": [
        ("hyperventilation", r"hyperventilation"),
        ("photic stimulation", r"photic stimulation"),
        ("artifact", r"\bartifact\b"),
        ("artefact", r"\bartefact\b"),
        ("eye opening", r"eye opening"),
        ("sleep", r"\bsleep\b"),
    ],
}


def _find_source_terms(pages):
    output = {}
    for group, patterns in EEG_TEXT_TERMS.items():
        hits = {}
        for page in pages:
            text = page["text"] or ""
            for label, pattern in patterns:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    hits.setdefault(label, []).append(page["page_number"])
        if hits:
            output[group] = [{"term": term, "pages": sorted(set(nums))} for term, nums in hits.items()]
    return output


def analyze_eeg_pdf_screening(pdf_path):
    """Research screening of PDF pages: text indexing + visual page outlier ranking.

    IsolationForest is fitted to page-image descriptors within this document. It is
    deliberately not presented as a trained clinical EEG classifier. Pages are
    rendered and analysed one at a time to limit memory use on small hosting plans.
    """
    pages = []
    feature_rows = []
    text_pages = []

    with fitz.open(pdf_path) as pdf:
        if pdf.is_encrypted:
            raise ValueError("PDF is password protected")
        page_count = len(pdf)
        if page_count == 0:
            raise ValueError("PDF contains no pages")
        if page_count > MAX_ANALYSIS_PAGES:
            raise ValueError(f"PDF has {page_count} pages; the current limit is {MAX_ANALYSIS_PAGES}")

        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text("text").strip()
            text_pages.append({"page_number": page_number, "text": page_text})

            rect = page.rect
            scale = min(1.5, 1700.0 / max(float(rect.width), float(rect.height), 1.0))
            scale = max(0.65, scale)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            features = _page_visual_features(pix.tobytes("png"))

            lower = page_text.lower()
            has_eeg_text_cue = bool(re.search(
                r"\b(eeg|montage|electroencephal|microvolt|\buv\b|\bµv\b|\bfp[12]\b|\bf[357z]\b|\bc[345z]\b|\bp[345z]\b|\bo[12]\b)\b",
                lower,
                flags=re.IGNORECASE,
            ))
            image_plot_like = (
                0.003 <= features["dark_pixel_fraction"] <= 0.35
                and features["active_row_fraction"] >= 0.20
                and features["active_column_fraction"] >= 0.35
                and (features["horizontal_adjacency"] >= 0.08 or features["column_density_variation"] >= 0.012)
            )
            waveform_candidate = bool(image_plot_like and (has_eeg_text_cue or features["active_column_fraction"] >= 0.60))
            if waveform_candidate:
                page_class = "Possible waveform/plot page — manual review"
            elif len(page_text) > 80:
                page_class = "Text/report page"
            elif features["dark_pixel_fraction"] > 0.002:
                page_class = "Image/scanned page — manual review"
            else:
                page_class = "Low-content page"

            pages.append({
                "page_number": page_number,
                "page_class": page_class,
                "waveform_like_candidate": waveform_candidate,
                "text_characters": len(page_text),
                **features,
                "visual_outlier": False,
                "visual_outlier_score": None,
            })
            feature_rows.append([
                features["dark_pixel_fraction"],
                features["active_row_fraction"],
                features["active_column_fraction"],
                features["horizontal_adjacency"],
                features["vertical_adjacency"],
                features["column_density_variation"],
                features["contrast_index"],
            ])

    ml_method = "IsolationForest page-visual outlier prioritisation (unsupervised; fitted only to pages in this PDF)"
    if len(feature_rows) >= 5:
        model = IsolationForest(n_estimators=100, contamination=0.15, random_state=42)
        matrix = np.asarray(feature_rows, dtype=float)
        labels = model.fit_predict(matrix)
        scores = model.decision_function(matrix)
        for idx, page in enumerate(pages):
            page["visual_outlier"] = bool(labels[idx] == -1)
            page["visual_outlier_score"] = round(float(-scores[idx]), 5)
    else:
        ml_method = "Visual feature screening only (fewer than 5 pages; no ML outlier ranking run)"

    terms = _find_source_terms(text_pages)
    waveform_pages = [p["page_number"] for p in pages if p["waveform_like_candidate"]]
    outlier_pages = [p["page_number"] for p in pages if p["visual_outlier"]]
    pages_with_text = sum(1 for p in text_pages if p["text"])
    detected_electrode_tokens = sorted({
        token.upper() for token in re.findall(
            r"\b(?:Fp[12]|F[357]|Fz|C[345]|Cz|P[345]|Pz|O[12]|T[345678]|A[12]|M[12])\b",
            " ".join(p["text"] for p in text_pages),
            flags=re.IGNORECASE,
        )
    })

    summary = (
        f"The PDF has {page_count} pages. {len(waveform_pages)} pages were visually prioritised as possible waveform/plot pages, "
        f"and {len(outlier_pages)} pages were unusual compared with other pages in this same document. "
        "These are page-review flags, not detected EEG abnormalities."
    )
    if detected_electrode_tokens:
        channel_note = f"Project target: {TARGET_CHANNEL_CONFIGURATION} channels. Electrode labels found in selectable text: {len(detected_electrode_tokens)} unique labels ({', '.join(detected_electrode_tokens[:24])}). This does not verify the number of channels or derivations."
    else:
        channel_note = f"Project target: {TARGET_CHANNEL_CONFIGURATION} channels. No electrode labels could be reliably counted from the selectable PDF text."

    clinician_note = (
        "CLINICIAN REVIEW REQUIRED — NO AUTONOMOUS EEG DIAGNOSIS GENERATED.\n\n"
        + summary + "\n\n" + channel_note + "\n\n"
        + "The machine-learning component only ranks visually unusual PDF pages. Visual-density peaks can arise from waveform deflections, printed text, labels, grid lines, or artifacts. "
        "Terms listed below are words located in the source report text; they are not independently confirmed EEG findings. "
        "This version does not confirm or exclude seizures, epileptiform discharges, focal/generalised slowing, or a normal EEG. "
        "A qualified clinical neurophysiologist/neurologist must inspect the entire original EEG and clinical context before signing an interpretation."
    )

    return {
        "page_count": page_count,
        "pages_with_text": pages_with_text,
        "waveform_like_pages": len(waveform_pages),
        "waveform_candidate_page_numbers": waveform_pages,
        "visual_outlier_pages": len(outlier_pages),
        "visual_outlier_page_numbers": outlier_pages,
        "source_text_terms": terms,
        "electrode_labels_from_text": detected_electrode_tokens,
        "channel_configuration": channel_note,
        "ml_method": ml_method,
        "summary": summary,
        "clinician_review_note": clinician_note,
        "pages": pages,
        "analysis_kind": "PDF visual screening and report-text indexing",
        "clinical_model_status": "No clinically trained/validated EEG classifier is installed in this project.",
    }


def create_screening_report(result, output_path):
    """Create a downloadable PDF summary without patient identifiers."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="NBSmall", parent=styles["BodyText"], fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="NBNote", parent=styles["BodyText"], fontSize=9, leading=12, textColor=colors.HexColor("#333333")))
    story = [
        Paragraph("NEUROBRIDGE EEG — PDF RESEARCH SCREENING REPORT", styles["Title"]),
        Spacer(1, 5 * mm),
        Paragraph("Research screening output · Not a diagnostic report", styles["Heading2"]),
        Paragraph(escape(result["summary"]), styles["BodyText"]),
        Spacer(1, 3 * mm),
        Paragraph(escape(result["channel_configuration"]), styles["BodyText"]),
        Spacer(1, 4 * mm),
        Paragraph("Machine-learning method", styles["Heading2"]),
        Paragraph(escape(result["ml_method"]), styles["BodyText"]),
        Paragraph("The unsupervised IsolationForest is fitted to visual page descriptors from this PDF only. It prioritises visually unusual pages; it is not trained to identify EEG pathology.", styles["NBNote"]),
        Spacer(1, 4 * mm),
        Paragraph("Page review table", styles["Heading2"]),
    ]
    data = [["Page", "Page type", "Visual outlier", "Ink %", "Density peaks (x%)"]]
    for page in result["pages"]:
        data.append([
            str(page["page_number"]),
            Paragraph(escape(page["page_class"]), styles["NBSmall"]),
            "Review" if page["visual_outlier"] else "—",
            f'{page["dark_pixel_fraction"] * 100:.2f}',
            ", ".join(str(x) for x in page["visual_density_peaks"][:6]) or "—",
        ])
    table = Table(data, repeatRows=1, colWidths=[12 * mm, 58 * mm, 25 * mm, 18 * mm, 48 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17344d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b8c5d1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f8")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.extend([Spacer(1, 5 * mm), Paragraph("Source report wording located", styles["Heading2"])])
    if result["source_text_terms"]:
        for group, hits in result["source_text_terms"].items():
            summary = "; ".join(f'{hit["term"]} (page {", ".join(str(p) for p in hit["pages"])})' for hit in hits)
            story.append(Paragraph(f"<b>{escape(group)}:</b> {escape(summary)}", styles["NBSmall"]))
    else:
        story.append(Paragraph("No selected EEG terms were located in selectable PDF text. Scanned images may not have a text layer.", styles["NBSmall"]))
    story.extend([Spacer(1, 5 * mm), Paragraph("Clinical review note", styles["Heading2"]), Paragraph(escape(result["clinician_review_note"]).replace("\n", "<br/>"), styles["NBNote"])])
    story.extend([Spacer(1, 4 * mm), Paragraph("Generated by NeuroBridge research software. This report must not be used to diagnose, treat, or exclude a neurological condition.", styles["NBSmall"])])
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=14 * mm, leftMargin=14 * mm, topMargin=14 * mm, bottomMargin=14 * mm)
    doc.build(story)


class Handler(BaseHTTPRequestHandler):
    server_version = "NeuroBridgeSimple/1.0"

    def log_message(self, fmt, *args):
        logging.info("%s - %s", self.address_string(), fmt % args)

    def send_data(self, code, data, content_type="application/json; charset=utf-8"):
        body = data if isinstance(data, bytes) else data.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self, limit=1024 * 1024):
        n = int(self.headers.get("Content-Length", "0"))
        if n > limit:
            raise ValueError("Request too large")
        raw = self.rfile.read(n)
        return json.loads(raw or b"{}")

    def authorised(self):
        token = self.headers.get("X-NB-Token", "")
        if not token:
            qs = urlparse(self.path).query
            for item in qs.split("&"):
                if item.startswith("token="):
                    from urllib.parse import unquote_plus
                    token = unquote_plus(item[6:])
                    break
        return bool(token and token in TOKENS)

    def require_auth(self):
        if not self.authorised():
            self.send_data(401, json_bytes({"detail": "Please sign in again."}))
            return False
        return True

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self.send_data(200, PAGE, "text/html; charset=utf-8")
        elif path == "/api/health":
            self.send_data(200, json_bytes({"ok": True, "service": APP_NAME, "storage": "temporary local storage", "max_file_mb": 50, "pdf_screening": True, "ml_method": "IsolationForest page-visual outlier prioritisation", "target_channels": TARGET_CHANNEL_CONFIGURATION}))
        elif path == "/api/files":
            if not self.require_auth(): return
            with db_connect() as con:
                rows = con.execute("SELECT id,name,size,mime,extension,uploaded_at FROM uploads ORDER BY uploaded_at DESC").fetchall()
            self.send_data(200, json_bytes({"files": [dict(r) for r in rows]}))
        elif path.startswith("/api/analyze/"):
            if not self.require_auth(): return
            file_id = unquote(path.rsplit("/", 1)[-1])
            with db_connect() as con:
                row = con.execute("SELECT * FROM uploads WHERE id=?", (file_id,)).fetchone()
            if not row:
                self.send_data(404, json_bytes({"detail": "Uploaded file not found."})); return
            if row["extension"].lower() != ".pdf":
                self.send_data(415, json_bytes({"detail": "PDF screening is available for PDF files."})); return
            pdf_path = UPLOAD_DIR / row["stored_name"]
            if not pdf_path.is_file():
                self.send_data(404, json_bytes({"detail": "The uploaded PDF is not present on this server. Please upload it again."})); return
            try:
                result = analyze_eeg_pdf_screening(pdf_path)
                analysis_id = uuid.uuid4().hex
                result["analysis_id"] = analysis_id
                result["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
                report_name = analysis_id + ".pdf"
                create_screening_report(result, ANALYSIS_DIR / report_name)
                with db_connect() as con:
                    con.execute(
                        "INSERT INTO analyses VALUES (?,?,?,?,?)",
                        (analysis_id, file_id, result["created_at"], json.dumps(result, ensure_ascii=False), report_name),
                    )
                self.send_data(200, json_bytes(result))
            except Exception as exc:
                logging.exception("PDF screening failed")
                self.send_data(400, json_bytes({"detail": "PDF screening failed: " + str(exc)}))
            return
        elif path.startswith("/api/analysis-report/"):
            if not self.require_auth(): return
            analysis_id = unquote(path.rsplit("/", 1)[-1])
            if not re.fullmatch(r"[a-f0-9]{32}", analysis_id):
                self.send_data(400, json_bytes({"detail": "Invalid report identifier."})); return
            with db_connect() as con:
                row = con.execute("SELECT report_name FROM analyses WHERE id=?", (analysis_id,)).fetchone()
            if not row:
                self.send_data(404, json_bytes({"detail": "Analysis report not found."})); return
            report_path = ANALYSIS_DIR / row["report_name"]
            if not report_path.is_file():
                self.send_data(404, json_bytes({"detail": "Report file is no longer on this server."})); return
            blob = report_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Content-Disposition", "attachment; filename=\"NeuroBridge_EEG_Screening_Report.pdf\"")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(blob)
            return
        elif path.startswith("/api/download/"):
            if not self.require_auth(): return
            file_id = unquote(path.rsplit("/", 1)[-1])
            with db_connect() as con:
                row = con.execute("SELECT * FROM uploads WHERE id=?", (file_id,)).fetchone()
            if not row:
                self.send_data(404, json_bytes({"detail": "File not found"})); return
            target = UPLOAD_DIR / row["stored_name"]
            if not target.is_file():
                self.send_data(404, json_bytes({"detail": "File is no longer on this server. Free hosting storage may be temporary."})); return
            blob = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", row["mime"])
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Content-Disposition", "attachment; filename=\"" + row["name"].replace('"', '') + "\"")
            self.end_headers()
            self.wfile.write(blob)
        else:
            self.send_data(404, json_bytes({"detail": "Not found"}))

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/login":
            try:
                data = self.read_json()
            except Exception:
                self.send_data(400, json_bytes({"detail": "Invalid login request"})); return
            if not ACCESS_CODE:
                self.send_data(503, json_bytes({"detail": "Owner must set NB_ACCESS_CODE in Render environment settings."})); return
            if not isinstance(data, dict) or data.get("access_code") != ACCESS_CODE:
                self.send_data(401, json_bytes({"detail": "Incorrect access code"})); return
            token = uuid.uuid4().hex
            TOKENS.add(token)
            self.send_data(200, json_bytes({"token": token, "message": "Login successful"}))
        elif path == "/api/upload":
            if not self.require_auth(): return
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type or "boundary=" not in content_type:
                self.send_data(400, json_bytes({"detail": "Choose a file before uploading."})); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_FILE_BYTES + 1024 * 1024:
                self.send_data(413, json_bytes({"detail": "Upload is empty or exceeds the 50 MB limit."})); return
            body = self.rfile.read(length)
            try:
                headers = ("Content-Type: " + content_type + "\r\nMIME-Version: 1.0\r\n\r\n").encode("utf-8")
                msg = BytesParser(policy=policy.default).parsebytes(headers + body)
                file_part = next((p for p in msg.iter_parts() if p.get_content_disposition() == "form-data" and p.get_param("name", header="content-disposition") == "file"), None)
                if file_part is None or not file_part.get_filename():
                    raise ValueError("No file was selected")
                original_name = safe_name(file_part.get_filename())
                ext = Path(original_name).suffix.lower()
                if ext not in ALLOWED_EXTENSIONS:
                    raise ValueError("Unsupported file type")
                blob = file_part.get_payload(decode=True) or b""
                if not blob or len(blob) > MAX_FILE_BYTES:
                    raise ValueError("File is empty or larger than 50 MB")
                if ext == ".pdf" and not blob.lstrip().startswith(b"%PDF-"):
                    raise ValueError("This file does not appear to be a valid PDF.")
            except Exception as exc:
                self.send_data(400, json_bytes({"detail": str(exc)})); return
            file_id = uuid.uuid4().hex
            stored_name = file_id + ext
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            (UPLOAD_DIR / stored_name).write_bytes(blob)
            mime = file_part.get_content_type() or mimetypes.guess_type(original_name)[0] or "application/octet-stream"
            uploaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            with db_connect() as con:
                con.execute("INSERT INTO uploads VALUES (?,?,?,?,?,?,?)", (file_id, original_name, stored_name, len(blob), mime, ext, uploaded_at))
            self.send_data(201, json_bytes({"id": file_id, "name": original_name, "size": len(blob), "uploaded_at": uploaded_at, "message": "Uploaded to the current server."}))
        else:
            self.send_data(404, json_bytes({"detail": "Not found"}))


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    host = "0.0.0.0"
    logging.info("Starting %s on %s:%s", APP_NAME, host, PORT)
    server = ThreadingHTTPServer((host, PORT), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
