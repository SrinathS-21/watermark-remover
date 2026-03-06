"""
Automatic Watermark Remover — Production Pipeline (v2)

Pipeline: YOLO11x Detection → SAM Pixel-Precise Mask → LaMa Inpainting
Uses:
  - corzent/yolo11x_watermark_detection (YOLO11x) for watermark detection
  - SAM ViT-B (Segment Anything) for pixel-precise mask refinement
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
import torch
from PIL import Image
from ultralytics import YOLO
from simple_lama_inpainting import SimpleLama
from segment_anything import sam_model_registry, SamPredictor

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
YOLO_MODEL = os.path.join(os.path.dirname(__file__), "models", "best.pt")
SAM_CHECKPOINT = os.path.join(os.path.dirname(__file__), "models", "sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = "vit_b"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_MASK_PADDING = 10
DEFAULT_MAX_PASSES = 2


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_yolo_model(device: str = "cpu") -> YOLO:
    """Load the YOLO11x watermark detection model."""
    logger.info("Loading YOLO11x watermark detection model...")
    if not os.path.exists(YOLO_MODEL):
        from huggingface_hub import hf_hub_download
        logger.info("Downloading YOLO11x weights from HuggingFace...")
        hf_hub_download("corzent/yolo11x_watermark_detection", "best.pt",
                        local_dir=os.path.join(os.path.dirname(__file__), "models"))
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


def create_mask_bbox(
    image_shape: tuple[int, int],
    detections: list[dict],
    padding: int = DEFAULT_MASK_PADDING,
) -> np.ndarray:
    """
    Create a binary inpainting mask from YOLO detections (bbox fallback).

    Returns:
        Binary mask (uint8): 0 = keep, 255 = inpaint
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    return mask


def create_mask_sam(
    sam_predictor: SamPredictor,
    image_rgb: np.ndarray,
    detections: list[dict],
    padding: int = DEFAULT_MASK_PADDING,
) -> np.ndarray:
    """
    Create a pixel-precise inpainting mask using SAM with bbox prompts.

    Each YOLO bbox is fed to SAM as a box prompt to get a tight segmentation
    mask around the watermark, then dilated by `padding` pixels.

    Returns:
        Binary mask (uint8): 0 = keep, 255 = inpaint
    """
    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    sam_predictor.set_image(image_rgb)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        input_box = np.array([x1, y1, x2, y2])
        masks, scores, _ = sam_predictor.predict(
            box=input_box[None, :],
            multimask_output=True,
        )
        # Pick the mask with highest score
        best_idx = int(np.argmax(scores))
        sam_mask = masks[best_idx].astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, sam_mask)

    # Dilate to cover edges
    if padding > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (padding * 2 + 1, padding * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

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
    sam_predictor: SamPredictor | None = None,
    max_passes: int = DEFAULT_MAX_PASSES,
) -> dict:
    """
    Full pipeline for a single image with multi-pass support:
        Pass 1: detect → SAM mask → inpaint
        Pass 2+: re-detect on inpainted result → inpaint residuals

    Returns a summary dict with detection count, output path, etc.
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
        image_name,
        len(detections),
        ", ".join(f'{d["label"]}({d["confidence"]:.2f})' for d in detections),
    )

    # 2. Load image
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    all_detections = list(detections)

    # 3. Create mask (SAM pixel-precise or bbox fallback)
    if sam_predictor is not None:
        mask = create_mask_sam(sam_predictor, img_array, detections, padding)
        logger.info("  [%s] SAM pixel-precise mask generated.", image_name)
    else:
        mask = create_mask_bbox((h, w), detections, padding)

    # 4. Inpaint
    result = inpaint_image(lama_model, image, mask)

    # 5. Multi-pass: re-detect on inpainted result and fix residuals
    for pass_num in range(2, max_passes + 1):
        result_array = np.array(result)
        # Save temp file for YOLO (it needs a file path)
        temp_path = os.path.join(output_dir, f"_temp_{image_name}")
        result.save(temp_path)
        residual_detections = detect_watermarks(yolo_model, temp_path, confidence)
        os.remove(temp_path)

        if not residual_detections:
            logger.info("  [%s] Pass %d: Clean — no residuals.", image_name, pass_num)
            break

        logger.info(
            "  [%s] Pass %d: Found %d residual watermark(s), re-inpainting.",
            image_name, pass_num, len(residual_detections),
        )
        all_detections.extend(residual_detections)

        if sam_predictor is not None:
            residual_mask = create_mask_sam(
                sam_predictor, result_array, residual_detections, padding
            )
        else:
            residual_mask = create_mask_bbox(
                (h, w), residual_detections, padding
            )
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
    use_sam: bool = True,
    max_passes: int = DEFAULT_MAX_PASSES,
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
        use_sam: use SAM for pixel-precise masks (else bbox fallback)
        max_passes: max detection-inpainting passes per image

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
            sam_predictor=sam_predictor, max_passes=max_passes,
        )
        results.append(result)

    elapsed = time.time() - start_time

    # Summary
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Automatic Watermark Remover v2 — YOLO11x + SAM + LaMa",
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
    parser.add_argument(
        "--no-sam", action="store_true",
        help="Disable SAM mask refinement (use bbox masks instead)",
    )
    parser.add_argument(
        "--max-passes", type=int, default=DEFAULT_MAX_PASSES,
        help=f"Max detection-inpainting passes per image (default: {DEFAULT_MAX_PASSES})",
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
        use_sam=not args.no_sam,
        max_passes=args.max_passes,
    )


if __name__ == "__main__":
    main()
