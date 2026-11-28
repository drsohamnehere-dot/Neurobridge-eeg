# NeuroBridge EEG Workbench v3.0.1 — deployment-corrected full-stack research prototype

This release upgrades the earlier browser-only app to a single-service full-stack application. FastAPI serves the web interface and API; SQLite stores study/file/analysis/model metadata; uploaded originals and trained model artifacts are stored on the server's persistent data directory.

## What changed

- **Visible upload progress:** per-file and batch percentage, bytes sent/total, transfer speed and approximate remaining time. A separate “Saving…” state appears after the browser has sent the bytes and before the backend confirms storage.
- **Large uploads:** 500 MB default per file (configurable with `NB_MAX_UPLOAD_MB`). The browser reports actual upload progress using `XMLHttpRequest.upload.onprogress`.
- **Server persistence:** original files are stored under `data/uploads`; database records are in `data/neurobridge.sqlite3`; model artifacts are under `data/models`. Docker Compose mounts `./data` as a persistent volume.
- **Study records:** save de-identified study code, date, montage, sampling rate, recording context and supplied report text.
- **File support:** CSV, TSV, TXT, JSON, EDF, BDF, PDF and common image formats. EDF/BDF parsing uses MNE. The EDF Analyzer offers 24-, 32- and 64-channel processing/display capacities, capped to the number of scalp EEG channels actually identified in the file; waveform preview supports up to 64 channels. A separate full-recording low-artifact epoch screening workflow ranks 2- or 4-second candidate windows, skips explicit BAD annotations and marked state transitions, reports candidate duration and stores its screening result. These engineering heuristics do not prove an epoch is artifact-free and do not diagnose clinical abnormalities. The main quantitative qEEG summary still covers the initial 120-second window. CSV/TSV analysis reports channel-level summary statistics and exploratory spectral-band power when a sampling rate can be estimated from a time column.
- **Page 6 — Clean Segment Finder:** a separate workflow page scans full EDF/BDF recordings in 2- or 4-second epochs, displays candidate low-artifact intervals and waveform review, and calculates per-state FFT rhythm indices and ratios from accepted candidate epochs. Outputs include Beta LF/HF indices, Alpha/Theta, Theta/Beta, and Theta/Beta LF. Rhythm boundaries, total spectrum range and FFT window are configurable to match an external analysis style. These are research comparison metrics, not clinical thresholds.
- **PDF review:** list PDF pages, preview rendered pages, and extract embedded PDF text when present.
- **Research ML baseline:** a StandardScaler + RBF-SVM model built from explicitly supplied `task_label` and `trial_id` data. Trials are split before evaluation. Saved models are versioned; adding CSV/TSV data marks older models stale. Retraining is an explicit user action.
- **Technical draft PDF:** export study metadata, file inventory and analyses with prominent research/clinical-safety caveats.
- **Access code:** backend-checked shared access code with expiring signed sessions (12 hours by default).

## Run locally with Docker

1. Install Docker Desktop (or Docker Engine + Compose) on a computer/server.
2. Copy `.env.example` to `.env` and set a private `NB_ACCESS_CODE` and a long random `NB_SECRET_KEY`.
3. From this folder, run:

   ```bash
   docker compose up --build -d
   ```

4. Open `http://localhost:8000` on that computer, or open `http://YOUR-COMPUTER-LAN-IP:8000` from another device on the same trusted network.
5. Enter the access code configured in `.env`.

To stop: `docker compose down`. Your files and database remain in `./data` because the folder is mounted as a volume. Do **not** delete the `data` folder if you want to retain stored studies.

## Put it online / use from iPad and phone

**Recommended:** deploy the full-stack Docker service using the included `render.yaml` Blueprint. It serves the interface and API from one HTTPS origin, reads the Render-provided `PORT`, and mounts a 10 GB persistent disk at `/data`. The Blueprint prompts for `NB_ACCESS_CODE` and generates `NB_SECRET_KEY`. Render persistent disks require a paid service plan. Do not remove the disk if you need to keep uploaded EEG files, the SQLite database, and saved models.

**Keep the current Netlify front end (optional):** this package contains `netlify_site/` with `index.html` at the publish root. Netlify's `netlify.toml` publishes that directory. Edit `netlify_site/config.js` to use the exact Render API URL ending in `/api`, and set Render's `NB_CORS_ORIGINS` to the exact Netlify origin. The browser sends file uploads directly to Render; do not proxy large upload bodies through Netlify. The Netlify site's current origin from the supplied screenshot is already placed in `render.yaml` as the initial CORS allow-list value.

For either route, use HTTPS, a private access code, institutional approval, restricted data access, and verified backups. Do not upload identifiable patient EEG data to a public or ephemeral filesystem. See `NETLIFY_404_AND_DEPLOYMENT_FIX.md` for exact steps. After deployment, open the Render or Netlify URL in Safari and use Share → Add to Home Screen.

## Upload and progress details

The percentage shown while sending is based on bytes reported by the browser. When it reaches 100%, the server may still be hashing and writing the file; the UI changes to “Saving…” until the API confirms the database record. If the connection is interrupted, the app reports an error; this release uses a standard single-request upload, not true resumable/chunk-offset uploads. Re-select the file to retry.

## Machine-learning data format

For supervised research task classification, upload CSV/TSV recordings with one row per sample, numeric EEG channel columns, and at least these metadata columns:

```csv
time_s,F3_uV,F4_uV,C3_uV,C4_uV,task_label,trial_id
0.000,1.2,-0.4,0.8,1.1,YES,session1_trial01
0.004,1.3,-0.3,0.7,1.0,YES,session1_trial01
...
```

- `task_label` must be the **predefined experimental cue/class actually presented to the participant**, not a guessed or retrospectively inferred thought.
- `trial_id` identifies one independent trial and is repeated on its samples.
- Use identical numeric channel columns across the files used for a model.
- Each class needs at least 3 independent trials in the current baseline implementation.
- The app aggregates features per trial and holds out a stratified trial subset for a first-pass test. This is an exploratory baseline only; one random split is not independent validation. A publishable study needs preregistered protocol, adequate sample size, subject/session-wise validation, artifact controls (including EOG/EMG where possible), chance comparison, confidence intervals and an untouched final test set.

The ML pipeline currently trains from CSV/TSV/TXT only. EDF/BDF are parsed for recording metadata and exploratory channel metrics; PDF/screenshots are stored and reviewed as documents/images. Those file types are **not silently converted into labelled speech trials**.

## Meaning of “updates itself”

New files and studies are saved to the backend automatically after upload. The web screen refreshes file/storage/model state every 15 seconds and after each upload. When a new tabular EEG file is uploaded, the old model is marked **stale**. It does not automatically absorb new labels: press **Train / retrain** to create a new version and see its held-out metrics. This avoids silently training on unreviewed or incorrectly labelled data.

## API overview

- `GET /api/health` — health check
- `POST /api/auth/login` — access-code login
- `GET/POST /api/studies` — list/save study metadata
- `POST /api/files` — upload a file with multipart form data
- `GET /api/files` — list stored file records
- `GET /api/files/{id}/download` — download original
- `POST /api/files/{id}/analyse` — parse and store technical analysis
- `GET /api/files/{id}/pdf-info` and `/pdf-page/{n}` — PDF page metadata and preview
- `POST /api/ml/train`, `GET /api/ml/models` — train and list model versions
- `POST /api/ml/predict/{model_id}/{file_id}` — constrained task-label predictions for a compatible CSV/TSV
- `GET /api/studies/{id}/report.pdf` — create technical review draft

All endpoints except `/api/health` and `/api/auth/login` require `Authorization: Bearer <token>`.

## Important scientific, medical and privacy limits

- Scalp EEG cannot reveal unrestricted private thoughts, memories, secrets or free-form sentences.
- A task classifier predicts from a small set of experiment-defined classes. It does not prove the participant's intended thought and it is not a validated communication aid.
- Routine clinical EEG, scans, images and written reports do not contain the required trial labels for imagined-speech ML.
- This software does not diagnose epilepsy, detect or exclude seizures, or make treatment decisions. It is not a validated medical device.
- The generated PDF is a technical draft only. A qualified neurologist/neurophysiologist must review the complete recording and clinical context.
- Use de-identified data. Obtain Institutional Ethics Committee approval, informed consent and institutional authorisation before collecting or storing human research data. Apply institution-approved retention, access control, encryption, backups and breach procedures.
- The access code is shared access, not a full multi-user role-based account system. Use a secure reverse proxy/HTTPS and stronger institutional identity controls before any real deployment.

## Folder structure

- `frontend/` — iPad/mobile-friendly interface served by the full-stack backend
- `netlify_site/` — static frontend bundle for Netlify; configure `config.js` with the Render API URL
- `render.yaml` — Render Docker Blueprint with a persistent data disk
- `server/main.py` — API, database, file handling, analysis, ML and PDF report generation
- `data/` — persistent runtime data (created automatically; do not include patient records in source control)
- `Dockerfile`, `docker-compose.yml` — deployment packaging

## Included demo data

`frontend/sample_data/demo_labelled_synthetic.csv` is an artificial signal dataset with four deliberately separable labels. It is useful only to check that file upload, trial aggregation and model training run end to end. The unusually structured demo is **not recorded human EEG**, and its accuracy must not be reported as scientific evidence. The `sample_template.csv` file shows the required column names.

## Reviewer-page bundle

In the Waveforms step, upload PNG/JPG/WEBP screenshots or add them by pasting into the page. In AI & ML review, open a PDF page list and tick the pages to include. Save up to **8 total images/PDF pages** against an active study. The exported technical draft includes those selected visual pages after the text summary.

## Prediction on a new trial file

After training, choose a saved model and upload/select a compatible new CSV/TSV. The prediction file needs `trial_id` and the same numeric channel columns used by the model; it does not need `task_label`. The returned class probabilities are experimental model outputs only.
