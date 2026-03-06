import argparse
import logging

from .config import DEFAULT_CONFIDENCE, DEFAULT_MASK_PADDING
from .pipeline import process_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
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
