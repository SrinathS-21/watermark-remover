import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from segment_anything import SamPredictor

from .config import (
    DEFAULT_CONFIDENCE, DEFAULT_MASK_PADDING,
    DEFAULT_MAX_PASSES, IMAGE_EXTENSIONS, SAM_CHECKPOINT,
)
from .detector import detect_watermarks
from .inpainter import inpaint_image
from .masker import create_mask_bbox, create_mask_sam
from .models import load_lama_model, load_sam_model, load_yolo11x_model, load_yolov8_model

logger = logging.getLogger(__name__)


def process_image(
    yolo11x_model,
    yolov8_model,
    lama_model,
    image_path: str,
    output_dir: str,
    confidence: float = DEFAULT_CONFIDENCE,
    padding: int = DEFAULT_MASK_PADDING,
    save_mask: bool = False,
    save_annotated: bool = False,
    sam_predictor: SamPredictor | None = None,
    max_passes: int = DEFAULT_MAX_PASSES,
) -> dict:
    """
    Dual-model two-pass pipeline for a single image:
        Pass 1 : YOLO11x detect (high recall) → SAM mask → LaMa inpaint
        Pass 2+: YOLOv8 detect (precision) on inpainted result → SAM mask → LaMa inpaint

    YOLO11x sweeps aggressively to catch all watermarks on the original image.
    YOLOv8 then checks the now-clean background for any surviving residuals with
    tighter bounding boxes, giving SAM a clean, unambiguous prompt.

    Returns a summary dict with detection count, output path, and details.
    """
    image_name = os.path.basename(image_path)

    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    all_detections = []
    result = image  # may be updated after pass 1 inpainting
    combined_mask = None  # accumulates masks from all passes

    # --- Pass 1: YOLO11x high-recall sweep ---
    pass1_detections = detect_watermarks(yolo11x_model, image_path, confidence)
    if pass1_detections:
        logger.info(
            "  [%s] Pass 1 (YOLO11x): Detected %d watermark(s): %s",
            image_name, len(pass1_detections),
            ", ".join(f'{d["label"]}({d["confidence"]:.2f})' for d in pass1_detections),
        )
        all_detections.extend(pass1_detections)
        if sam_predictor is not None:
            mask = create_mask_sam(sam_predictor, img_array, pass1_detections, padding)
            logger.info("  [%s] Pass 1: SAM pixel-precise mask generated.", image_name)
        else:
            mask = create_mask_bbox((h, w), pass1_detections, padding)
        combined_mask = mask
        result = inpaint_image(lama_model, image, mask)
    else:
        logger.info(
            "  [%s] Pass 1 (YOLO11x): Nothing detected — running YOLOv8 fallback check.",
            image_name,
        )

    # --- Pass 2+: YOLOv8 precision check (runs on inpainted result OR original) ---
    for pass_num in range(2, max_passes + 1):
        result_array = np.array(result)
        cur_h, cur_w = result_array.shape[:2]
        temp_path = os.path.join(output_dir, f"_temp_{image_name}")
        result.save(temp_path)
        residual_detections = detect_watermarks(yolov8_model, temp_path, confidence)
        os.remove(temp_path)

        if not residual_detections:
            logger.info("  [%s] Pass %d (YOLOv8): Clean — no watermarks.", image_name, pass_num)
            break

        logger.info(
            "  [%s] Pass %d (YOLOv8): Found %d watermark(s), inpainting.",
            image_name, pass_num, len(residual_detections),
        )
        all_detections.extend(residual_detections)

        if sam_predictor is not None:
            residual_mask = create_mask_sam(
                sam_predictor, result_array, residual_detections, padding
            )
        else:
            residual_mask = create_mask_bbox((cur_h, cur_w), residual_detections, padding)

        # Merge into combined mask (resize residual_mask to match combined_mask if needed)
        if combined_mask is None:
            combined_mask = residual_mask
        else:
            if residual_mask.shape == combined_mask.shape:
                combined_mask = cv2.bitwise_or(combined_mask, residual_mask)
            else:
                residual_mask_resized = cv2.resize(
                    residual_mask, (combined_mask.shape[1], combined_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                combined_mask = cv2.bitwise_or(combined_mask, residual_mask_resized)

        result = inpaint_image(lama_model, result, residual_mask)

    # Neither model found anything — copy original
    if not all_detections:
        logger.info("  [%s] No watermarks detected by either model — copying original.", image_name)
        out_path = os.path.join(output_dir, image_name)
        result.save(out_path)
        return {"image": image_name, "detections": 0, "output": out_path}

    # --- Save outputs ---
    out_path = os.path.join(output_dir, image_name)
    result.save(out_path)

    if save_mask and combined_mask is not None:
        mask_dir = os.path.join(output_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        Image.fromarray(combined_mask).save(os.path.join(mask_dir, image_name))

    if save_annotated:
        annotated_dir = os.path.join(output_dir, "annotated")
        os.makedirs(annotated_dir, exist_ok=True)
        annotated = img_array.copy()
        for det in all_detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label_text = f'{det["label"]} {det["confidence"]:.2f}'
            cv2.putText(
                annotated, label_text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
            )
        Image.fromarray(annotated).save(os.path.join(annotated_dir, image_name))

    return {
        "image": image_name,
        "detections": len(all_detections),
        "output": out_path,
        "details": all_detections,
    }


def process_batch(
    input_dir: str,
    output_dir: str,
    confidence: float = DEFAULT_CONFIDENCE,
    padding: int = DEFAULT_MASK_PADDING,
    save_mask: bool = False,
    save_annotated: bool = False,
    device: str = "cpu",
    use_sam: bool = True,
    max_passes: int = DEFAULT_MAX_PASSES,
) -> list[dict]:
    """Process all images in a directory with the YOLO11x + YOLOv8 + SAM + LaMa pipeline.

    Pass 1 uses YOLO11x for high-recall watermark detection on the original image.
    Pass 2 uses YOLOv8 for precision residual detection on the inpainted result.

    Args:
        input_dir: directory containing input images
        output_dir: directory to save cleaned images
        confidence: YOLO detection confidence threshold
        padding: mask padding / SAM dilation radius in pixels
        save_mask: also save the binary masks
        save_annotated: also save images with detection bboxes drawn
        device: 'cpu' or 'cuda'
        use_sam: use SAM for pixel-precise masks (False = bbox fallback)
        max_passes: max detection-inpainting passes per image

    Returns:
        List of result dicts from process_image()
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    images = sorted(
        f for f in input_path.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file()
    )
    if not images:
        logger.error("No images found in: %s", input_dir)
        sys.exit(1)

    logger.info("Found %d images in: %s", len(images), input_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    yolo11x_model = load_yolo11x_model(device)
    yolov8_model = load_yolov8_model(device)
    lama_model = load_lama_model()
    sam_predictor = None
    if use_sam and os.path.exists(SAM_CHECKPOINT):
        sam_predictor = load_sam_model(device)
    elif use_sam:
        logger.warning(
            "SAM checkpoint not found at %s — falling back to bbox masks.",
            SAM_CHECKPOINT,
        )

    results = []
    total = len(images)
    start_time = time.time()

    for idx, img_path in enumerate(images, 1):
        logger.info("[%d/%d] Processing: %s", idx, total, img_path.name)
        result = process_image(
            yolo11x_model, yolov8_model, lama_model, str(img_path), output_dir,
            confidence=confidence, padding=padding,
            save_mask=save_mask, save_annotated=save_annotated,
            sam_predictor=sam_predictor, max_passes=max_passes,
        )
        results.append(result)

    elapsed = time.time() - start_time
    total_detections = sum(r["detections"] for r in results)
    images_with_wm = sum(1 for r in results if r["detections"] > 0)

    logger.info("=" * 60)
    logger.info("BATCH COMPLETE (YOLO11x + YOLOv8 + %s + LaMa)",
                "SAM" if sam_predictor else "bbox")
    logger.info("  Total images processed : %d", total)
    logger.info("  Images with watermarks : %d", images_with_wm)
    logger.info("  Total watermarks found : %d", total_detections)
    logger.info("  Max passes per image   : %d", max_passes)
    logger.info("  Time elapsed           : %.1fs", elapsed)
    logger.info("  Output directory       : %s", output_dir)
    logger.info("=" * 60)

    return results
