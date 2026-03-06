"""
watermark_remover — Automatic watermark detection and removal.

Public API:
    load_yolo_model()   — load the YOLOv8 detection model
    load_lama_model()   — load the LaMa inpainting model
    process_image()     — run the full pipeline on a single image
    process_batch()     — run the full pipeline on a directory of images
"""

from .models import load_lama_model, load_yolo_model
from .pipeline import process_batch, process_image

__all__ = [
    "load_yolo_model",
    "load_lama_model",
    "process_image",
    "process_batch",
]
