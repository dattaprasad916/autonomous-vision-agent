from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import cv2
import numpy as np
import time
import os
import uuid
from collections import Counter

from app.detector import ModelRegistry
from app.opencv_objects import discover_objects
from app.database import insert_detection

router = APIRouter()
templates = Jinja2Templates(directory="templates")

model_registry = ModelRegistry()


# ---------------- HOME ----------------
@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# ---------------- HYBRID DETECTION ----------------
@router.post("/detect", response_class=HTMLResponse)
async def detect_ui(
    request: Request,
    file: UploadFile = File(...),
    model: str = "yolov8n"
):
    start_time = time.time()

    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    os.makedirs("static/results", exist_ok=True)

    # Save raw image
    raw_path = f"static/results/raw_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(raw_path, image)

    # YOLO detection (known objects)
    yolo_boxes, detections = model_registry.detect(image, model)

    # OpenCV discovery (unknown objects)
    unknown_boxes = discover_objects(image)

    # Draw YOLO boxes (GREEN)
    for x1, y1, x2, y2, label, conf in yolo_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{label} {conf:.2f}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # Draw OpenCV boxes (BLUE)
    for x1, y1, x2, y2 in unknown_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Save processed image
    processed_path = f"static/results/hybrid_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(processed_path, image)

    inference_time_ms = int((time.time() - start_time) * 1000)
    counts = dict(Counter(d["label"] for d in detections))

    # Log analytics
    for d in detections:
        insert_detection(
            model=model,
            label=d["label"],
            confidence=d["confidence"],
            inference_time=inference_time_ms
        )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "raw_image": "/" + raw_path,
            "processed_image": "/" + processed_path,
            "counts": counts,
            "time_ms": inference_time_ms,
            "model_used": model,
            "known_count": len(yolo_boxes),
            "unknown_count": len(unknown_boxes)
        }
    )


# ---------------- ANALYTICS ----------------
@router.get("/analytics", response_class=HTMLResponse)
async def analytics(request: Request):
    from app.database import fetch_all
    from collections import defaultdict

    data = fetch_all()

    label_counts = Counter()
    model_counts = Counter()
    confidence_map = defaultdict(list)
    inference_times = []

    for model, label, conf, inf_time in data:
        label_counts[label] += 1
        model_counts[model] += 1
        confidence_map[label].append(conf)
        inference_times.append(inf_time)

    avg_conf = {
        k: round(sum(v) / len(v), 3)
        for k, v in confidence_map.items()
    }

    avg_time = int(sum(inference_times) / len(inference_times)) if inference_times else 0

    return templates.TemplateResponse(
        "analytics.html",
        {
            "request": request,
            "label_counts": label_counts,
            "model_counts": model_counts,
            "avg_conf": avg_conf,
            "avg_time": avg_time
        }
    )
