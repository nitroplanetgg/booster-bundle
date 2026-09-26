"""
booster_api.py

Minimal HTTP API wrapping the trained booster-bundle detector, so other
services (a web frontend, a mobile app, another backend) can get
detections over a network call instead of running the Python scripts
directly. Loads the model once at startup and reuses it for every
request -- reloading ONNX weights per-request would be slow.

Setup:
    pip install fastapi uvicorn python-multipart

Run (either works):
    python booster_api.py --weights runs/detect/train/weights/best.onnx --catalog catalog.json
    uvicorn booster_api:app --host 0.0.0.0 --port 8000
        (reads config from BOOSTER_WEIGHTS / BOOSTER_CONF / BOOSTER_IMGSZ /
         BOOSTER_DEVICE / BOOSTER_CATALOG env vars when launched this way,
         since there's no argparse in that path -- see configure() below)

catalog.json is the product catalog (series -> products, each with an
"id" slug and a "tcg_id"). Each detection's class_name (from your trained
model, i.e. whatever bundle_id folder name you labeled with) is looked up
against BOTH the "id" and "tcg_id" fields, so it works regardless of
which one your bundle_id folders happen to match. Detections with no
catalog match still come back, just with catalog fields set to null.

Endpoints:
    GET  /health             -> {"status": "ok", "classes": [...]}
    POST /detect              -> multipart image upload, returns JSON detections (with catalog metadata)
    POST /detect/annotated    -> multipart image upload, returns the annotated image (PNG)

Try it:
    curl -X POST -F "file=@photo.jpg" http://127.0.0.1:8000/detect
    curl -X POST -F "file=@photo.jpg" http://127.0.0.1:8000/detect/annotated --output out.png
    # interactive docs at http://127.0.0.1:8000/docs
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from ultralytics import YOLO

app = FastAPI(title="Booster Bundle Detector")

# Populated by configure() before the server starts handling requests, so
# request handlers never need to touch disk or reload the model.
model: YOLO | None = None
CONF_THRESHOLD = 0.25
IMGSZ = 640
DEVICE = "cpu"
CATALOG_INDEX: dict = {}  # {"id" or "tcg_id": {series_id, series_name, set_id, set_name, image_url}}


def build_catalog_index(catalog: dict) -> dict:
    index = {}
    for series in catalog.values():
        series_id = series["id"]
        series_name = series["name"]
        for product in series.get("products", []):
            entry = {
                "seriesId": series_id,
                "series": series_name,
                "setId": product["tcg_id"],
                "set": product["name"],
                "date": product["release_date"],
                "image_url": product.get("imageUrl"),

            }
            index[product["id"]] = entry
            if product.get("tcg_id"):
                index[product["tcg_id"]] = entry
    return index


def load_catalog(catalog_path: str) -> None:
    global CATALOG_INDEX
    path = Path(catalog_path)
    if not path.exists():
        print(f"Warning: catalog not found at {path} -- detections will have no series/set metadata")
        CATALOG_INDEX = {}
        return
    with open(path) as f:
        raw = json.load(f)
    CATALOG_INDEX = build_catalog_index(raw)
    print(f"Loaded catalog: {len(CATALOG_INDEX)} lookup keys from {path}")


def configure(weights_path: str, conf: float = 0.25, imgsz: int = 640, device: str = "cpu", catalog_path: str = "catalog.json"):
    global model, CONF_THRESHOLD, IMGSZ, DEVICE
    weights = Path(weights_path)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found at {weights}")
    model = YOLO(str(weights))
    CONF_THRESHOLD = conf
    IMGSZ = imgsz
    DEVICE = device
    load_catalog(catalog_path)


@app.on_event("startup")
def _startup():
    # Covers running via `uvicorn booster_api:app` directly (no argparse
    # path in that case), reading the same config from env vars instead.
    if model is None:
        configure(
            weights_path=os.environ.get("BOOSTER_WEIGHTS", "model/booster_bundles.onnx"),
            conf=float(os.environ.get("BOOSTER_CONF", 0.25)),
            imgsz=int(os.environ.get("BOOSTER_IMGSZ", 640)),
            device=os.environ.get("BOOSTER_DEVICE", "cpu"),
            catalog_path=os.environ.get("BOOSTER_CATALOG", "data/boosterbundle.json"),
        )


def _decode_image(file_bytes: bytes) -> np.ndarray:
    # cv2.imdecode gives a BGR array, matching cv2.imread -- the same
    # convention used everywhere else in this pipeline (labeling,
    # augmentation), and what Ultralytics expects for raw ndarray input.
    arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return image


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "classes": list(model.names.values())}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    image = _decode_image(await file.read())

    results = model.predict(source=image, conf=CONF_THRESHOLD, imgsz=IMGSZ, device=DEVICE, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        catalog_entry = CATALOG_INDEX.get(class_name)  # tries both "id" and "tcg_id" keys
        detections.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(float(box.conf[0]), 4),
            "box_xyxy": [round(v, 1) for v in box.xyxy[0].tolist()],
            "seriesId": catalog_entry["seriesId"] if catalog_entry else None,
            "series": catalog_entry["series"] if catalog_entry else None,
            "setId": catalog_entry["setId"] if catalog_entry else None,
            "set": catalog_entry["set"] if catalog_entry else None,
            "date": catalog_entry["date"] if catalog_entry else None,
            "imageURL": catalog_entry["image_url"] if catalog_entry else None,
            "type": "4" # booster bundle type
        })

    return JSONResponse({"filename": file.filename, "detections": detections})


@app.post("/detect/annotated")
async def detect_annotated(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    image = _decode_image(await file.read())

    results = model.predict(source=image, conf=CONF_THRESHOLD, imgsz=IMGSZ, device=DEVICE, verbose=False)
    annotated = results[0].plot()  # BGR numpy array with boxes drawn

    ok, encoded = cv2.imencode(".png", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image")
    return Response(content=encoded.tobytes(), media_type="image/png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="model/booster_bundles.onnx")
    parser.add_argument("--catalog", default="data/boosterbundle.json", help="Product catalog JSON for series/set metadata lookup")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Set env vars so the /startup handler (which also covers the
    # `uvicorn booster_api:app` launch path) picks up the same config
    # uniformly, regardless of how the process was started.
    os.environ["BOOSTER_WEIGHTS"] = args.weights
    os.environ["BOOSTER_CATALOG"] = args.catalog
    os.environ["BOOSTER_CONF"] = str(args.conf)
    os.environ["BOOSTER_IMGSZ"] = str(args.imgsz)
    os.environ["BOOSTER_DEVICE"] = args.device

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()