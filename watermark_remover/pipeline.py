import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import DEFAULT_CONFIDENCE, DEFAULT_MASK_PADDING, IMAGE_EXTENSIONS
from .detector import detect_watermarks
from .inpainter import inpaint_image
from .masker import create_mask
from .models import load_lama_model, load_yolo_model

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
) -> dict:
    """
    Full pipeline for a single image: detect → mask → inpaint → save.

    Returns a summary dict with detection count, output path, etc.
    """
    image_name = os.path.basename(image_path)

    # 1. Detect
    detections = detect_watermarks(yolo_model, image_path, confidence)

    if not detections:
        logger.info("  [%s] No watermarks detected — copying original.", image_name)
        original = Image.open(image_path).convert("RGB")
        out_path = os.path.join(output_dir, image_name)
        original.save(out_path)
        return {"image": image_name, "detections": 0, "output": out_path}

    logger.info(
        "  [%s] Detected %d watermark(s): %s",
        image_name,
        len(detections),
        ", ".join(f'{d["label"]}({d["confidence"]:.2f})' for d in detections),
    )

    # 2. Load image
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    # 3. Create mask
    mask = create_mask((h, w), detections, padding)

    # 4. Inpaint
    result = inpaint_image(lama_model, image, mask)

    # 5. Save outputs
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
        for det in detections:
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
        "detections": len(detections),
        "output": out_path,
        "details": detections,
    }


def process_batch(
    input_dir: str,
    output_dir: str,
    confidence: float = DEFAULT_CONFIDENCE,
    padding: int = DEFAULT_MASK_PADDING,
    save_mask: bool = False,
    save_annotated: bool = False,
    device: str = "cpu",
) -> list[dict]:
    """
    Process all images in a directory.

    Args:
        input_dir: directory containing input images
        output_dir: directory to save cleaned images
        confidence: YOLO detection confidence threshold
        padding: mask padding in pixels around each detection
        save_mask: also save the binary masks
        save_annotated: also save images with detection bboxes drawn
        device: 'cpu' or 'cuda'

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

    yolo_model = load_yolo_model()
    lama_model = load_lama_model()

    results = []
    total = len(images)
    start_time = time.time()

    for idx, img_path in enumerate(images, 1):
        logger.info("[%d/%d] Processing: %s", idx, total, img_path.name)
        result = process_image(
            yolo_model, lama_model, str(img_path), output_dir,
            confidence=confidence, padding=padding,
            save_mask=save_mask, save_annotated=save_annotated,
        )
        results.append(result)

    elapsed = time.time() - start_time
    total_detections = sum(r["detections"] for r in results)
    images_with_wm = sum(1 for r in results if r["detections"] > 0)

    logger.info("=" * 60)
    logger.info("BATCH COMPLETE")
    logger.info("  Total images processed : %d", total)
    logger.info("  Images with watermarks : %d", images_with_wm)
    logger.info("  Total watermarks found : %d", total_detections)
    logger.info("  Time elapsed           : %.1fs", elapsed)
    logger.info("  Output directory       : %s", output_dir)
    logger.info("=" * 60)

    return results
