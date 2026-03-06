import logging

from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from simple_lama_inpainting import SimpleLama

from .config import YOLO_REPO, YOLO_FILENAME

logger = logging.getLogger(__name__)


def load_yolo_model() -> YOLO:
    """Download (if needed) and load the YOLOv8 watermark detection model."""
    logger.info("Loading YOLOv8 watermark detection model...")
    model_path = hf_hub_download(repo_id=YOLO_REPO, filename=YOLO_FILENAME)
    model = YOLO(model_path)
    logger.info("YOLOv8 model loaded. Classes: %s", model.names)
    return model


def load_lama_model() -> SimpleLama:
    """Load the LaMa inpainting model (downloads weights on first use)."""
    logger.info("Loading LaMa inpainting model...")
    lama = SimpleLama()
    logger.info("LaMa model loaded.")
    return lama
