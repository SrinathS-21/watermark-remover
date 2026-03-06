import os

# ---------------------------------------------------------------------------
# Supported image formats
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ---------------------------------------------------------------------------
# YOLO11x model (local path — auto-downloaded if missing)
# ---------------------------------------------------------------------------
YOLO_MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "best.pt")
YOLO_REPO = "corzent/yolo11x_watermark_detection"
YOLO_FILENAME = "best.pt"

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
