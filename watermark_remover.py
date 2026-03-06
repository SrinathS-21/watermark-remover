"""
Automatic Watermark Remover — Production Pipeline

Pipeline: YOLO Detection → Mask Generation → LaMa Inpainting
Uses:
  - mnemic/watermarks_yolov8 (YOLOv8) for watermark detection
  - big-lama (LaMa) via simple-lama-inpainting for watermark removal
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from simple_lama_inpainting import SimpleLama

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
YOLO_REPO = "mnemic/watermarks_yolov8"
YOLO_FILENAME = "watermarks_s_yolov8_v1.pt"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_MASK_PADDING = 10


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_yolo_model() -> YOLO:
    """Download and load the YOLOv8 watermark detection model."""
    logger.info("Loading YOLOv8 watermark detection model...")
    model_path = hf_hub_download(repo_id=YOLO_REPO, filename=YOLO_FILENAME)
    model = YOLO(model_path)
    logger.info("YOLOv8 model loaded. Classes: %s", model.names)
    return model


def load_lama_model() -> SimpleLama:
    """Load the LaMa inpainting model (downloads on first use)."""
    logger.info("Loading LaMa inpainting model...")
    lama = SimpleLama()
    logger.info("LaMa model loaded.")
    return lama


# ---------------------------------------------------------------------------
# Core Pipeline Functions
# ---------------------------------------------------------------------------
def detect_watermarks(
    model: YOLO,
    image_path: str,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[dict]:
    """
    Run YOLOv8 inference on a single image.

    Returns list of detections:
        [{"bbox": [x1,y1,x2,y2], "confidence": float, "label": str}, ...]
    """
    results = model.predict(image_path, conf=confidence, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            label = model.names[cls_id]
            detections.append({
                "bbox": coords,
                "confidence": conf,
                "label": label,
            })
    return detections


def create_mask(
    image_shape: tuple[int, int],
    detections: list[dict],
    padding: int = DEFAULT_MASK_PADDING,
) -> np.ndarray:
    """
    Create a binary inpainting mask from YOLO detections.

    Args:
        image_shape: (height, width) of the original image
        detections: list of detection dicts with "bbox" key
        padding: pixels to expand each bbox for cleaner inpainting edges

    Returns:
        Binary mask (uint8): 0 = keep, 255 = inpaint
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        # Apply padding and clamp to image bounds
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    return mask


def inpaint_image(
    lama: SimpleLama,
    image: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    """
    Run LaMa inpainting to remove watermark regions.

    Args:
        lama: loaded SimpleLama model
        image: original PIL image
        mask: binary mask (uint8), 255 = regions to inpaint

    Returns:
        Inpainted PIL image
    """
    mask_pil = Image.fromarray(mask)
    result = lama(image, mask_pil)
    return result


# ---------------------------------------------------------------------------
# Single Image Processing
# ---------------------------------------------------------------------------
def process_image(
    yolo_model: YOLO,
    lama_model: SimpleLama,
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


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------
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

    # Collect images
    images = sorted(
        f for f in input_path.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file()
    )
    if not images:
        logger.error("No images found in: %s", input_dir)
        sys.exit(1)

    logger.info("Found %d images in: %s", len(images), input_dir)

    # Create output dirs
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    yolo_model = load_yolo_model()
    lama_model = load_lama_model()

    # Process
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

    # Summary
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Automatic Watermark Remover — YOLO Detection + LaMa Inpainting",
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input directory containing images",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for cleaned images",
    )
    parser.add_argument(
        "--confidence", "-c", type=float, default=DEFAULT_CONFIDENCE,
        help=f"YOLO detection confidence threshold (default: {DEFAULT_CONFIDENCE})",
    )
    parser.add_argument(
        "--padding", "-p", type=int, default=DEFAULT_MASK_PADDING,
        help=f"Mask padding in pixels around detections (default: {DEFAULT_MASK_PADDING})",
    )
    parser.add_argument(
        "--save-mask", action="store_true",
        help="Save binary masks to output/masks/",
    )
    parser.add_argument(
        "--save-annotated", action="store_true",
        help="Save annotated images (with detection boxes) to output/annotated/",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Device for inference (default: cpu)",
    )
    args = parser.parse_args()

    process_batch(
        input_dir=args.input,
        output_dir=args.output,
        confidence=args.confidence,
        padding=args.padding,
        save_mask=args.save_mask,
        save_annotated=args.save_annotated,
        device=args.device,
    )


if __name__ == "__main__":
    main()
