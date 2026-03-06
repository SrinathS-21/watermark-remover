import os

# ---------------------------------------------------------------------------
# Supported image formats
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ---------------------------------------------------------------------------
# YOLO11x model — Pass 1 (high recall: catches all watermarks)
# ---------------------------------------------------------------------------
YOLO11X_MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "best.pt")
YOLO11X_REPO = "corzent/yolo11x_watermark_detection"
YOLO11X_FILENAME = "best.pt"

# ---------------------------------------------------------------------------
# YOLOv8 model — Pass 2 (precision: targets surviving residuals on clean bg)
# ---------------------------------------------------------------------------
YOLOV8_MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "watermarks_s_yolov8_v1.pt")
YOLOV8_REPO = "mnemic/watermarks_yolov8"
YOLOV8_FILENAME = "watermarks_s_yolov8_v1.pt"

# ---------------------------------------------------------------------------
# SAM ViT-B checkpoint (local path)
# ---------------------------------------------------------------------------
SAM_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "models", "sam_vit_b_01ec64.pth"
)
SAM_MODEL_TYPE = "vit_b"

# ---------------------------------------------------------------------------
# Pipeline defaults
# ---------------------------------------------------------------------------
DEFAULT_CONFIDENCE = 0.25
DEFAULT_MASK_PADDING = 10
DEFAULT_MAX_PASSES = 2
