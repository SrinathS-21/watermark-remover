import argparse
import logging

from .config import DEFAULT_CONFIDENCE, DEFAULT_MASK_PADDING, DEFAULT_MAX_PASSES
from .pipeline import process_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automatic Watermark Remover v3 — YOLO11x + YOLOv8 + SAM + LaMa",
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input directory containing images")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for cleaned images")
    parser.add_argument("--confidence", "-c", type=float, default=DEFAULT_CONFIDENCE,
                        help=f"YOLO confidence threshold (default: {DEFAULT_CONFIDENCE})")
    parser.add_argument("--padding", "-p", type=int, default=DEFAULT_MASK_PADDING,
                        help=f"Mask padding / SAM dilation in pixels (default: {DEFAULT_MASK_PADDING})")
    parser.add_argument("--save-mask", action="store_true",
                        help="Save binary masks to output/masks/")
    parser.add_argument("--save-annotated", action="store_true",
                        help="Save annotated images to output/annotated/")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Inference device (default: cpu)")
    parser.add_argument("--no-sam", action="store_true",
                        help="Disable SAM — use bbox masks instead")
    parser.add_argument("--max-passes", type=int, default=DEFAULT_MAX_PASSES,
                        help=f"Max inpainting passes per image (default: {DEFAULT_MAX_PASSES})")
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
