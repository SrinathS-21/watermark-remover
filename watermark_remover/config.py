import os

# ---------------------------------------------------------------------------
# Supported image formats
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ---------------------------------------------------------------------------
# YOLOv8 model (local path — auto-downloaded if missing)
# ---------------------------------------------------------------------------
YOLO_MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "watermarks_s_yolov8_v1.pt")
YOLO_REPO = "mnemic/watermarks_yolov8"
YOLO_FILENAME = "watermarks_s_yolov8_v1.pt"

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
