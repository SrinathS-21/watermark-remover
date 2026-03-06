"""
watermark_remover — Automatic watermark detection and removal (v3).

Dual-model pipeline:
    Pass 1: YOLO11x (high recall) → SAM Pixel-Precise Mask → LaMa Inpainting
    Pass 2: YOLOv8  (precision)   → SAM Pixel-Precise Mask → LaMa Inpainting

Public API:
    load_yolo11x_model() — load the YOLO11x detection model (pass 1)
    load_yolov8_model()  — load the YOLOv8 detection model (pass 2)
    load_sam_model()     — load the SAM ViT-B predictor
    load_lama_model()    — load the LaMa inpainting model
    process_image()      — run the full dual-pass pipeline on a single image
    process_batch()      — run the full pipeline on a directory of images
"""

from .models import load_lama_model, load_sam_model, load_yolo11x_model, load_yolov8_model
from .pipeline import process_batch, process_image

__all__ = [
    "load_yolo11x_model",
    "load_yolov8_model",
    "load_sam_model",
    "load_lama_model",
    "process_image",
    "process_batch",
]
