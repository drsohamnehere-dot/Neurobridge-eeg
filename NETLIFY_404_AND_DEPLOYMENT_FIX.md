# NeuroBridge v3.0.1 — Netlify 404 and full-stack deployment fix

## Why the old Netlify deployment showed 404

The previous ZIP had this structure:

    neurobridge_v3/frontend/index.html

Netlify was being given the outer folder as the site root. Its publish directory therefore did not contain `index.html`. Netlify requires an `index.html` in the configured publish directory.

This release fixes the packaging and adds `netlify.toml`, which sets the publish directory to `netlify_site/`. That directory has `index.html`, CSS, JavaScript, icon, manifest, and `_redirects` at its root.

## Important hosting choice

The full application uses FastAPI, SQLite, saved EEG files, and saved ML model files. Netlify is static hosting; it does not run this FastAPI/Docker server or provide the persistent local disk required by the current SQLite/file-storage design.

Use Render (or another Docker host with a persistent disk) for the backend. This release contains a Render Blueprint (`render.yaml`) and a Dockerfile that respects the platform-provided `PORT`.

For files above 10 MB, the Netlify-hosted frontend sends uploads directly from the browser to the Render API using `NB_API_BASE`. Do not proxy large EEG uploads through Netlify rewrite rules: Netlify proxy rewrites have a short request timeout. Direct browser-to-Render requests also require the exact Netlify origin in `NB_CORS_ORIGINS`.

## Deployment (recommended: keep your existing Netlify site)

### A. Deploy the full-stack backend

1. Extract this ZIP. The project files (`render.yaml`, `Dockerfile`, `server/`, `frontend/`, etc.) are at the extracted project root. Do not upload the ZIP as a single file inside your Git repository.
2. Create a private GitHub repository and upload the extracted project contents to the repository root.
3. In Render, choose **New → Blueprint**, connect that repository, and let Render read `render.yaml`.
4. During setup, set `NB_ACCESS_CODE` to a private code only you/authorised reviewers know. Render generates `NB_SECRET_KEY` automatically.
5. The Blueprint creates a Docker web service and a 10 GB persistent disk mounted at `/data`. Persistent disks require a paid Render service plan. Do not remove the disk if you need to retain uploaded files, the SQLite database, and model files.
6. After deployment, copy the service URL, for example `https://neurobridge-eeg-workbench.onrender.com`.
7. In the Render service's environment settings, verify `NB_CORS_ORIGINS` is exactly `https://subtle-melomakarona-a94805.netlify.app` (no trailing slash). If your Netlify domain differs, replace it with the exact origin and redeploy.
8. Test `https://YOUR-RENDER-SERVICE.onrender.com/api/health`. It should return JSON indicating the API is healthy.

### B. Fix and redeploy the Netlify frontend

1. Open `netlify_site/config.js` in this extracted project.
2. Replace the placeholder with your actual Render API URL, keeping `/api` at the end. Example:

       window.NB_API_BASE = "https://neurobridge-eeg-workbench.onrender.com/api";

3. Commit/push that change to GitHub.
4. In Netlify, connect this repository (or update the existing site's repository) and leave the Base directory empty. The included `netlify.toml` sets **Publish directory** to `netlify_site` and the build command to blank.
5. Trigger a new production deploy. In the Deploy File Explorer, confirm that `index.html`, `app.js`, `config.js`, and `styles.css` are at the published root.
6. Open the Netlify URL. The dashboard should show **Backend online** after you enter the app access code.

### C. If using manual Netlify Drop instead of Git

Upload the *contents of the `netlify_site` folder*, not the parent ZIP/project folder. Before uploading, set the real Render API URL in `netlify_site/config.js`. The published root must contain `index.html` and `config.js`.

## Alternative: use Render only

The Render service also serves the app UI and API on the same origin. You can simply open the Render service URL and skip Netlify entirely. In that case `frontend/config.js` already points to `/api` and no CORS setting is needed.

## Netlify project visibility

The screenshot shows the Netlify project marked **Private**. Private projects are only available to your Netlify team/invited users. Keep this private while testing with sensitive material. Only make the front-end public if you intentionally want anyone with the URL to see the login page; public visibility does not make the backend suitable for identifiable patient data.

## Data protection and clinical limits

- Test with synthetic or de-identified EEG first. Do not upload identifiable patient records until your institution has approved the hosting location, access controls, retention, and backups.
- The ML workflow classifies predefined, protocol-labelled experimental task classes only. It does not decode unrestricted thoughts or diagnose epilepsy/seizures.
- The generated PDF is a technical review draft, not a signed clinical report.
