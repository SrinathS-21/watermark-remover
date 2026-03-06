import logging

from ultralytics import YOLO

from .config import DEFAULT_CONFIDENCE

logger = logging.getLogger(__name__)


def detect_watermarks(
    model: YOLO,
    image_path: str,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[dict]:
    """
    Run YOLOv8 inference on a single image.

    Returns list of detections:
        [{"bbox": [x1, y1, x2, y2], "confidence": float, "label": str}, ...]
    """
    results = model.predict(image_path, conf=confidence, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            label = model.names[cls_id]
            detections.append({
                "bbox": coords,
                "confidence": conf,
                "label": label,
            })
    return detections
