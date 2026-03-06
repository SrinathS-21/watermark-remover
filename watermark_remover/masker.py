import cv2
import numpy as np
from segment_anything import SamPredictor

from .config import DEFAULT_MASK_PADDING


def create_mask_bbox(
    image_shape: tuple[int, int],
    detections: list[dict],
    padding: int = DEFAULT_MASK_PADDING,
) -> np.ndarray:
    """
    Build a binary mask from YOLO bounding boxes (SAM fallback).

    Args:
        image_shape: (height, width) of the source image
        detections: list of detection dicts with a "bbox" key ([x1,y1,x2,y2])
        padding: pixels to expand each bbox

    Returns:
        uint8 array (h, w): 0 = keep, 255 = inpaint
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
    Build a pixel-precise mask using SAM with YOLO bbox prompts.

    Each bbox is fed to SAM as a box prompt; the highest-scoring mask is
    selected per detection, then all masks are union-merged and dilated by
    `padding` pixels for clean inpainting edges.

    Args:
        sam_predictor: loaded SamPredictor instance
        image_rgb: HxWx3 uint8 numpy array (RGB)
        detections: list of detection dicts with a "bbox" key
        padding: dilation radius in pixels

    Returns:
        uint8 array (h, w): 0 = keep, 255 = inpaint
    """
    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    sam_predictor.set_image(image_rgb)

    for det in detections:
        input_box = np.array(det["bbox"])
        masks, scores, _ = sam_predictor.predict(
            box=input_box[None, :],
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        sam_mask = masks[best_idx].astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, sam_mask)

    if padding > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (padding * 2 + 1, padding * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask
