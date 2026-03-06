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
from .models import load_lama_model, load_sam_model, load_yolo_model

logger = logging.getLogger(__name__)


def process_image(
    yolo_model,
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
    Full multi-pass pipeline for a single image:
        Pass 1 : detect → SAM mask (or bbox) → LaMa inpaint
        Pass 2+ : re-detect residuals on inpainted result → inpaint again

    Returns a summary dict with detection count, output path, and details.
    """
    image_name = os.path.basename(image_path)

    # 1. Initial detection
    detections = detect_watermarks(yolo_model, image_path, confidence)
    if not detections:
        logger.info("  [%s] No watermarks detected — copying original.", image_name)
        original = Image.open(image_path).convert("RGB")
        out_path = os.path.join(output_dir, image_name)
        original.save(out_path)
        return {"image": image_name, "detections": 0, "output": out_path}

    logger.info(
        "  [%s] Pass 1: Detected %d watermark(s): %s",
        image_name, len(detections),
        ", ".join(f'{d["label"]}({d["confidence"]:.2f})' for d in detections),
    )

    # 2. Load image
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    all_detections = list(detections)

    # 3. Create mask — SAM pixel-precise or bbox fallback
    if sam_predictor is not None:
        mask = create_mask_sam(sam_predictor, img_array, detections, padding)
        logger.info("  [%s] SAM pixel-precise mask generated.", image_name)
    else:
        mask = create_mask_bbox((h, w), detections, padding)

    # 4. LaMa inpaint
    result = inpaint_image(lama_model, image, mask)

    # 5. Multi-pass: re-detect and fix residuals
    for pass_num in range(2, max_passes + 1):
        result_array = np.array(result)
        temp_path = os.path.join(output_dir, f"_temp_{image_name}")
        result.save(temp_path)
        residual_detections = detect_watermarks(yolo_model, temp_path, confidence)
        os.remove(temp_path)

        if not residual_detections:
            logger.info("  [%s] Pass %d: Clean — no residuals.", image_name, pass_num)
            break

        logger.info(
            "  [%s] Pass %d: Found %d residual(s), re-inpainting.",
            image_name, pass_num, len(residual_detections),
        )
        all_detections.extend(residual_detections)

        if sam_predictor is not None:
            residual_mask = create_mask_sam(
                sam_predictor, result_array, residual_detections, padding
            )
        else:
            residual_mask = create_mask_bbox((h, w), residual_detections, padding)

        mask = cv2.bitwise_or(mask, residual_mask)
        result = inpaint_image(lama_model, result, residual_mask)

    # 6. Save outputs
    out_path = os.path.join(output_dir, image_name)
    result.save(out_path)

    if save_mask:
        mask_dir = os.path.join(output_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        Image.fromarray(mask).save(os.path.join(mask_dir, image_name))

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
    """
    Process all images in a directory with the YOLO11x + SAM + LaMa pipeline.

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
    yolo_model = load_yolo_model(device)
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
            yolo_model, lama_model, str(img_path), output_dir,
            confidence=confidence, padding=padding,
            save_mask=save_mask, save_annotated=save_annotated,
            sam_predictor=sam_predictor, max_passes=max_passes,
        )
        results.append(result)

    elapsed = time.time() - start_time
    total_detections = sum(r["detections"] for r in results)
    images_with_wm = sum(1 for r in results if r["detections"] > 0)

    logger.info("=" * 60)
    logger.info("BATCH COMPLETE (YOLO11x + %s + LaMa)",
                "SAM" if sam_predictor else "bbox")
    logger.info("  Total images processed : %d", total)
    logger.info("  Images with watermarks : %d", images_with_wm)
    logger.info("  Total watermarks found : %d", total_detections)
    logger.info("  Max passes per image   : %d", max_passes)
    logger.info("  Time elapsed           : %.1fs", elapsed)
    logger.info("  Output directory       : %s", output_dir)
    logger.info("=" * 60)

    return results
