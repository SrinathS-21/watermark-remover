import cv2
import numpy as np

from .config import DEFAULT_MASK_PADDING


def create_mask(
    image_shape: tuple[int, int],
    detections: list[dict],
    padding: int = DEFAULT_MASK_PADDING,
) -> np.ndarray:
    """
    Build a binary inpainting mask from YOLO detections.

    Args:
        image_shape: (height, width) of the source image
        detections: list of detection dicts with a "bbox" key ([x1,y1,x2,y2])
        padding: pixels to expand each bbox for cleaner inpainting edges

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
