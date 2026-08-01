"""Person detection and crowd density estimation.

Wraps a YOLO model. Loads the baseline pretrained weights and, if present,
your fine-tuned weights, so the API can switch between them for comparison.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0

BASELINE_WEIGHTS = "yolov8n.pt"
FINETUNED_WEIGHTS = Path(__file__).parent / "weights" / "finetuned.pt"

# Provisional thresholds. Recalibrate once you know real platform area.
DENSITY_THRESHOLDS = {"low": 8, "moderate": 20}

_models: Dict[str, YOLO] = {}


def get_model(name: str = "finetuned") -> YOLO:
    """Load a model once and keep it in memory."""
    if name not in _models:
        if name == "finetuned" and FINETUNED_WEIGHTS.exists():
            _models[name] = YOLO(str(FINETUNED_WEIGHTS))
        else:
            _models[name] = YOLO(BASELINE_WEIGHTS)
    return _models[name]


def available_models() -> List[str]:
    models = ["baseline"]
    if FINETUNED_WEIGHTS.exists():
        models.append("finetuned")
    return models


def classify_density(count: int) -> str:
    if count <= DENSITY_THRESHOLDS["low"]:
        return "Low"
    if count <= DENSITY_THRESHOLDS["moderate"]:
        return "Moderate"
    return "Crowded"


def zone_breakdown(boxes: List[Dict], image_width: int) -> List[Dict]:
    """Split the frame into left / centre / right and count people in each."""
    names = ["Left end", "Centre", "Right end"]
    counts = [0, 0, 0]

    for box in boxes:
        centre_x = (box["x1"] + box["x2"]) / 2
        index = min(int(centre_x / (image_width / 3)), 2)
        counts[index] += 1

    return [
        {"zone": names[i], "count": counts[i], "density": classify_density(counts[i] * 3)}
        for i in range(3)
    ]


def detect(image: np.ndarray, model_name: str = "finetuned", confidence: float = 0.35) -> Dict:
    """Run detection on one image and return counts, boxes and density."""
    model = get_model(model_name)
    results = model.predict(
        source=image,
        conf=confidence,
        classes=[PERSON_CLASS_ID],
        verbose=False,
    )[0]

    boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes.append(
            {
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
                "confidence": round(float(box.conf[0]), 3),
            }
        )

    height, width = image.shape[:2]
    count = len(boxes)

    return {
        "model": model_name,
        "confidence_threshold": confidence,
        "count": count,
        "density": classify_density(count),
        "zones": zone_breakdown(boxes, width),
        "boxes": boxes,
        "image_size": {"width": width, "height": height},
    }
