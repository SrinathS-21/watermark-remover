import numpy as np
from PIL import Image
from simple_lama_inpainting import SimpleLama


def inpaint_image(
    lama: SimpleLama,
    image: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    """
    Run LaMa inpainting to fill watermark regions.

    Args:
        lama: loaded SimpleLama instance
        image: original PIL image (RGB)
        mask: uint8 array, 255 = regions to inpaint

    Returns:
        Inpainted PIL image
    """
    mask_pil = Image.fromarray(mask)
    return lama(image, mask_pil)
