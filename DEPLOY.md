# Deploying to Vercel

Vercel auto-detects a `Dockerfile.vercel` at your project root, builds it
as a container image, pushes it to Vercel Container Registry, and serves
it from a Function on Fluid Compute -- same git-push workflow, preview
deployments, and autoscaling as any other Vercel project. No `vercel.json`
is required for this.

---

## Step 1: Project folder layout

```
booster-vercel\
    Dockerfile.vercel
    requirements.txt
    booster_api.py
    data\
        boosterbundle.json
    model\
        booster_bundles.onnx   <- copy this in manually from your training output
```

Everything below assumes you're `cd`'d into this folder.

## Step 2: Install the Vercel CLI (optional, for local testing / manual deploys)

```powershell
npm i -g vercel
```

## Step 3: Test the container locally (optional but recommended)

```powershell
docker build -f Dockerfile.vercel -t booster-api .
docker run -p 8000:8000 -e PORT=8000 booster-api
```

```powershell
curl.exe -X POST -F "file=@photo.jpg" http://127.0.0.1:8000/detect
```

## Step 4: Deploy

Either connect the repo in the Vercel dashboard (Import Project -> pick
the repo -> deploy on every push), or deploy straight from the CLI:

```powershell
vercel deploy --prod
```

Vercel will build `Dockerfile.vercel` remotely and give you a URL like
`https://booster-vercel.vercel.app/`.

## Step 5: Test the live deployment

```powershell
curl.exe -X POST -F "file=@photo.jpg" https://booster-vercel.vercel.app/detect
curl.exe -X POST -F "file=@photo.jpg" https://booster-vercel.vercel.app/detect/annotated --output out.png
```

The first request after a period of inactivity will be slower (cold
start -- loading torch/ultralytics). Requests while the container stays
warm will be much faster.

## Overriding config without rebuilding

`BOOSTER_WEIGHTS`, `BOOSTER_CATALOG`, `BOOSTER_CONF`, `BOOSTER_IMGSZ`, and
`BOOSTER_DEVICE` are all read from environment variables. Set them under
Project Settings -> Environment Variables in the Vercel dashboard (or
with `vercel env add`) if you ever need to point at different weights or
tune the confidence threshold per-environment, without touching the
Dockerfile.

## Redeploying after you retrain

Copy the new `best.onnx` into `model/booster_bundles.onnx` (overwriting
the old one), commit, and push -- Vercel rebuilds and redeploys
automatically. With the CLI: `vercel deploy --prod` again from the
project folder.

## If something goes wrong

- **Build fails on `docker build`** -- usually a flaky network blip
  pulling torch (it's a big wheel). Just rerun.
- **`libGL.so.1: cannot open shared object file` at runtime** -- means
  `libgl1`/`libglib2.0-0` didn't get installed; check that step wasn't
  removed from `Dockerfile.vercel`.
- **502 / import error on the deployed function** -- check the Vercel
  dashboard's Function logs for the actual Python traceback (`vercel
  logs` from the CLI also works).
- **Slow on every request, not just the first** -- the container isn't
  staying warm between requests, which is expected for low-traffic
  functions; there's nothing Lambda-Provisioned-Concurrency-equivalent
  needed here unless traffic grows and it becomes a real problem.