import logging
import os

import torch
from ultralytics import YOLO
from simple_lama_inpainting import SimpleLama
from segment_anything import sam_model_registry, SamPredictor

from .config import (
    YOLO_MODEL, YOLO_REPO, YOLO_FILENAME,
    SAM_CHECKPOINT, SAM_MODEL_TYPE,
)

logger = logging.getLogger(__name__)


def load_yolo_model(device: str = "cpu") -> YOLO:
    """Load the YOLO11x watermark detection model (auto-downloads if missing)."""
    logger.info("Loading YOLO11x watermark detection model...")
    if not os.path.exists(YOLO_MODEL):
        from huggingface_hub import hf_hub_download
        logger.info("Downloading YOLO11x weights from HuggingFace...")
        models_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "models")
        )
        hf_hub_download(YOLO_REPO, YOLO_FILENAME, local_dir=models_dir)
    model = YOLO(YOLO_MODEL)
    logger.info("YOLO11x model loaded. Classes: %s", model.names)
    return model


def load_sam_model(device: str = "cpu") -> SamPredictor:
    """Load SAM ViT-B for pixel-precise mask generation."""
    logger.info("Loading SAM ViT-B model...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    logger.info("SAM model loaded.")
    return predictor


def load_lama_model() -> SimpleLama:
    """Load the LaMa inpainting model (downloads weights on first use)."""
    logger.info("Loading LaMa inpainting model...")
    lama = SimpleLama()
    logger.info("LaMa model loaded.")
    return lama
