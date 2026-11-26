#!/usr/bin/env python3
"""NeuroBridge one-file starter for Render.

A small, dependency-free upload and study-file dashboard.
Research/testing use only. It stores uploaded originals and metadata; it does NOT
interpret EEGs, diagnose abnormalities, or train a clinical machine-learning model.
On Render Free, local files and SQLite data are temporary and may be lost on restart
or instance replacement. Do not store identifiable patient data here.
"""

import os
import re
import json
import uuid
import sqlite3
import mimetypes
import logging
import fitz
from pathlib import Path
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote
from email.parser import BytesParser
from email import policy

APP_NAME = "NEUROBRIDGE"
PORT = int(os.environ.get("PORT", "10000"))
ACCESS_CODE = os.environ.get("NB_ACCESS_CODE", "")
DATA_DIR = Path(os.environ.get("NB_DATA_DIR", "/tmp/neurobridge_data"))
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "neurobridge.sqlite3"
MAX_FILE_BYTES = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".tsv", ".txt", ".edf", ".bdf", ".pdf", ".png", ".jpg", ".jpeg"}
TOKENS = set()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEUROBRIDGE EEG Workbench</title>
<style>
:root{color-scheme:dark;--bg:#07111e;--panel:#102238;--line:#29445e;--muted:#9fb0c3;--cyan:#65dce2;--white:#edf5ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--white);font:16px system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:22px max(18px,calc((100% - 1000px)/2));border-bottom:1px solid var(--line);background:#091827}header h1{margin:0;font-size:23px;letter-spacing:2px}header p{margin:5px 0 0;color:var(--muted);font-size:12px;letter-spacing:1px}
main{max-width:1000px;margin:24px auto;padding:0 16px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;margin-bottom:18px}.notice{background:#302817;border:1px solid #69542b;color:#f4e5bd;border-radius:12px;padding:13px;font-size:13px;line-height:1.5;margin-bottom:18px}
label{display:block;font-size:13px;color:#c2d1e1;margin:12px 0 6px}input{width:100%;padding:13px;border-radius:10px;border:1px solid var(--line);background:#081522;color:white;font:inherit}button{background:var(--cyan);color:#06202a;border:0;border-radius:10px;padding:12px 17px;font-weight:700;font-size:15px;cursor:pointer}button:disabled{opacity:.5;cursor:not-allowed}.secondary{background:#20364c;color:var(--white);border:1px solid var(--line)}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.drop{border:2px dashed #41627d;border-radius:14px;padding:28px 16px;text-align:center;background:#0a1a2a;margin:15px 0}.drop.drag{border-color:var(--cyan);background:#102b3a}.muted{color:var(--muted);font-size:13px}.hidden{display:none!important}.status{margin-top:12px;color:var(--cyan);white-space:pre-wrap;font-size:14px}.error{color:#ffb0aa}.progress{height:13px;border-radius:20px;background:#071522;overflow:hidden;margin:10px 0;border:1px solid var(--line)}.bar{height:100%;width:0;background:var(--cyan);transition:width .12s}.file{border-top:1px solid var(--line);padding:12px 0;display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap}.pill{font-size:11px;padding:4px 8px;border:1px solid var(--line);border-radius:30px;color:#c7d8e8}.small{font-size:12px;color:var(--muted)}
@media(max-width:600px){header{padding:18px}main{margin:15px auto}.panel{padding:15px}}
</style></head><body>
<header><h1>🧠 NEUROBRIDGE</h1><p>EEG FILE WORKBENCH · SIMPLE RESEARCH STARTER</p></header>
<main>
<div class="notice"><b>Research/testing version.</b> This starter stores uploaded files and displays upload progress. It does not interpret EEG signals, diagnose abnormalities, or provide clinical advice. Render Free storage is temporary. Do not upload identifiable patient information.</div>
<section class="panel" id="loginPanel"><h2>Private application access</h2><p class="muted">Enter the access code set by the owner in the Render environment settings.</p><label for="code">Access code</label><input id="code" type="password" autocomplete="current-password" placeholder="Your private access code"><div class="row" style="margin-top:14px"><button id="loginBtn" onclick="login()">Open application</button></div><div id="loginStatus" class="status"></div></section>
<section class="panel hidden" id="appPanel"><div class="row" style="justify-content:space-between"><div><h2>Upload EEG files</h2><p class="muted">Choose a CSV, TSV, TXT, EDF, BDF, PDF or waveform image. Maximum 50 MB per file.</p></div><button class="secondary" onclick="logout()">Sign out</button></div>
<div class="drop" id="drop"><div style="font-size:34px">⇧</div><b>Choose a file or drop it here</b><p class="muted">Your upload progress will appear below.</p><input id="fileInput" type="file" accept=".csv,.tsv,.txt,.edf,.bdf,.pdf,.png,.jpg,.jpeg"></div>
<div id="uploadArea" class="hidden"><div class="row" style="justify-content:space-between"><b id="uploadName">Uploading…</b><b id="pct">0%</b></div><div class="progress"><div class="bar" id="bar"></div></div><div class="small" id="uploadDetail">Waiting to upload</div></div><div id="uploadStatus" class="status"></div>
<div class="row" style="margin-top:18px;justify-content:space-between"><h3 style="margin:0">Files stored by this running server</h3><button class="secondary" onclick="loadFiles()">Refresh list</button></div><div id="files" class="muted" style="margin-top:10px">Loading…</div>
</section>
<section class="panel hidden" id="healthPanel"><b>Server status</b><div id="healthText" class="muted">Connected</div></section>
</main>
<script>
let token='';const $=id=>document.getElementById(id);
function status(id,msg,err=false){$(id).textContent=msg;$(id).className='status'+(err?' error':'')}
async function login(){const code=$('code').value;if(!code){status('loginStatus','Enter your access code.',true);return}$('loginBtn').disabled=true;try{const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({access_code:code})});const d=await r.json();if(!r.ok)throw Error(d.detail||'Login failed');token=d.token;$('loginPanel').classList.add('hidden');$('appPanel').classList.remove('hidden');$('healthPanel').classList.remove('hidden');status('healthText','Connected to NeuroBridge backend.');loadFiles()}catch(e){status('loginStatus',e.message,true)}finally{$('loginBtn').disabled=false}}
function logout(){token='';$('appPanel').classList.add('hidden');$('healthPanel').classList.add('hidden');$('loginPanel').classList.remove('hidden');$('code').value=''}
async function loadFiles(){try{const r=await fetch('/api/files',{headers:{'X-NB-Token':token}});const d=await r.json();if(!r.ok)throw Error(d.detail||'Could not load files');if(!d.files.length){$('files').textContent='No uploads yet.';return}$('files').innerHTML=d.files.map(f=>`<div class="file"><div><b>${esc(f.name)}</b><div class="small">${(f.size/1048576).toFixed(2)} MB · ${esc(f.uploaded_at)}</div></div><span class="pill">${esc(f.extension.toUpperCase())}</span><a style="color:#65dce2" href="/api/download/${encodeURIComponent(f.id)}?token=${encodeURIComponent(token)}">Download</a></div>`).join('')}catch(e){$('files').textContent=e.message}}
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function upload(file){if(!file)return;if(file.size>50*1024*1024){status('uploadStatus','This file is larger than 50 MB. Please choose a smaller file.',true);return}const ext='.'+file.name.split('.').pop().toLowerCase();if(!['.csv','.tsv','.txt','.edf','.bdf','.pdf','.png','.jpg','.jpeg'].includes(ext)){status('uploadStatus','This file type is not supported.',true);return}$('uploadArea').classList.remove('hidden');$('uploadName').textContent=file.name;$('pct').textContent='0%';$('bar').style.width='0%';$('uploadDetail').textContent='Starting upload…';status('uploadStatus','');const fd=new FormData();fd.append('file',file);const x=new XMLHttpRequest();x.open('POST','/api/upload');x.setRequestHeader('X-NB-Token',token);x.upload.onprogress=e=>{if(e.lengthComputable){const p=Math.round(e.loaded/e.total*100);$('pct').textContent=p+'%';$('bar').style.width=p+'%';$('uploadDetail').textContent=(e.loaded/1048576).toFixed(2)+' MB of '+(e.total/1048576).toFixed(2)+' MB uploaded'}};x.onload=()=>{try{const d=JSON.parse(x.responseText);if(x.status<200||x.status>=300)throw Error(d.detail||'Upload failed');$('pct').textContent='100%';$('bar').style.width='100%';$('uploadDetail').textContent='Upload complete · saved on current server';status('uploadStatus','File uploaded successfully.');loadFiles()}catch(e){status('uploadStatus',e.message,true)}};x.onerror=()=>status('uploadStatus','Network error. The upload did not finish.',true);x.send(fd)}
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


def safe_name(name):
    name = Path(name.replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    return (name or "upload.bin")[:120]
    
def render_eeg_pdf(pdf_path, zoom=2.0, max_pages=120):
    import fitz

    images = []

    with fitz.open(pdf_path) as pdf:
        if pdf.is_encrypted:
            raise ValueError("PDF is password protected")

        if len(pdf) == 0:
            raise ValueError("Empty PDF")

        if len(pdf) > max_pages:
            raise ValueError("Too many pages")

        for number, page in enumerate(pdf):
            pix = page.get_pixmap(
                matrix=fitz.Matrix(zoom, zoom),
                alpha=False
            )

            images.append({
                "page_number": number + 1,
                "image": pix.tobytes("png"),
                "width": pix.width,
                "height": pix.height
            })

    return images

def extract_eeg_pdf_text(pdf_path, max_pages=120):
    import fitz

    extracted_pages = []

    with fitz.open(pdf_path) as pdf:
        if pdf.is_encrypted:
            raise ValueError("PDF is password protected")

        if len(pdf) == 0:
            raise ValueError("Empty PDF")

        if len(pdf) > max_pages:
            raise ValueError("Too many pages")

        for number, page in enumerate(pdf):
            page_text = page.get_text("text").strip()

            extracted_pages.append({
                "page_number": number + 1,
                "text": page_text
            })

    return extracted_pages


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
            self.send_data(200, json_bytes({"ok": True, "service": APP_NAME, "storage": "temporary local storage", "max_file_mb": 50}))
        elif path == "/api/files":
            if not self.require_auth(): return
            with db_connect() as con:
                rows = con.execute("SELECT id,name,size,mime,extension,uploaded_at FROM uploads ORDER BY uploaded_at DESC").fetchall()
            self.send_data(200, json_bytes({"files": [dict(r) for r in rows]}))
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
    init_db()
    host = "0.0.0.0"
    logging.info("Starting %s on %s:%s", APP_NAME, host, PORT)
    server = ThreadingHTTPServer((host, PORT), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
