# Automatic Watermark Remover

> A deep learning pipeline for automatically detecting and removing watermarks from product images.
>
> **Version:** 2.0.0 &nbsp;|&nbsp; **Python:** ≥ 3.10 &nbsp;|&nbsp; **Package Manager:** uv

---

## Table of Contents

1. [Overview](#1-overview)
2. [Key Features](#2-key-features)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Phase 1 — Basic Pipeline (YOLOv8 + LaMa)](#4-phase-1--basic-pipeline-yolov8--lama)
5. [Phase 2 — Upgraded Pipeline (YOLO11x + SAM + Multi-Pass)](#5-phase-2--upgraded-pipeline-yolo11x--sam--multi-pass)
6. [Installation & Setup](#6-installation--setup)
7. [Usage](#7-usage)
8. [CLI Reference](#8-cli-reference)
9. [Project Structure](#9-project-structure)
10. [Performance & Hardware](#10-performance--hardware)

---

## 1. Overview

Watermarks are commonly embedded in images to protect intellectual property. In legitimate workflows — such as dataset cleaning, image restoration, or content editing — there is a need to automatically detect and remove them at scale.

This project provides a fully automated, end-to-end pipeline that:

- Detects watermarks using a trained object detection model
- Generates precise masks for the detected regions
- Removes watermarks using a high-quality deep inpainting model
- Saves cleaned images with optional debug outputs (masks, annotated detections)

The pipeline was built in two phases. Phase 1 established a working baseline; Phase 2 upgraded every stage for higher accuracy and cleaner output.

---

## 2. Key Features

| Feature | Description |
|---------|-------------|
| **Smart Detection** | YOLO11x trained specifically for watermark detection — mAP@50: 0.900, 1280px input |
| **Pixel-Precise Masks** | SAM (Segment Anything Model) traces the exact watermark boundary instead of using rectangles |
| **High-Quality Inpainting** | LaMa (Large Mask Inpainting) reconstructs the background using surrounding context |
| **Multi-Pass Removal** | After inpainting, re-runs detection to catch any residual traces |
| **Batch Processing** | Processes entire folders automatically |
| **Hardware Flexible** | Runs on CPU or GPU (`--device cuda`) |
| **Debug Outputs** | Optional binary masks and annotated detection images |

---

## 3. Pipeline Architecture

The system follows a 5-stage deep learning pipeline.

### Stage Overview

```
Input Image
    │
    ▼
┌──────────────────────────────┐
│  Stage 1: Detection          │
│  YOLO11x (1280px, mAP 0.900) │  → bounding boxes + confidence scores
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Stage 2: Mask Generation    │
│  SAM ViT-B (pixel-precise)   │  → binary mask tracing watermark boundary
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Stage 3: Inpainting Pass 1  │
│  LaMa big-lama               │  → watermark region reconstructed
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Stage 4: Residual Check     │
│  YOLO11x re-runs on result   │  → clean? done : run SAM + LaMa again
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Stage 5: Output             │
│  Cleaned image               │  + optional masks/ + annotated/
└──────────────────────────────┘
```

### Stage Details

**Stage 1 — Watermark Detection**
YOLO11x scans the image and predicts bounding boxes around watermark regions. Each detection includes a bounding box, confidence score, and class label.

**Stage 2 — Mask Generation**
SAM ViT-B takes the YOLO bounding boxes as prompts and generates pixel-precise segmentation masks — tracing the exact watermark shape rather than a rectangle. Morphological dilation adds a small boundary buffer. Falls back to rectangular bbox masks if SAM is unavailable (`--no-sam`).

**Stage 3 — Image Inpainting**
LaMa reconstructs the masked region using surrounding context: removes watermark pixels, fills the region, and preserves textures and patterns.

**Stage 4 — Residual Check**
After inpainting, YOLO11x runs again on the result. If any watermark is still detected, SAM + LaMa repeat for that region. Controlled by `--max-passes`.

**Stage 5 — Output**
Saves the cleaned image. Optionally saves binary masks (`--save-mask`) and annotated detection images (`--save-annotated`).

---

## 4. Phase 1 — Basic Pipeline (YOLOv8 + LaMa)

### Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| Detection | `mnemic/watermarks_yolov8` (YOLOv8-small) | 22.5 MB, 640px input |
| Masking | `cv2.rectangle()` + 10px padding | Rectangular bbox mask |
| Inpainting | LaMa `big-lama` | 196 MB, single pass |
| Package manager | `uv` | |

### Pipeline

```
Input Image → YOLOv8 Detection → Rectangular Mask → LaMa Inpainting → Output
```

### Results (16 images, CPU)

| Metric | Value |
|--------|-------|
| Images with watermarks detected | 7 |
| Total watermarks found | 11 |
| Processing time | 17.4s (~1.1s/image) |

**Key limitation:** Rectangular masks include non-watermark pixels (bbox corners), forcing LaMa to inpaint slightly more than needed. The 640px input also misses small watermarks.

---

## 5. Phase 2 — Upgraded Pipeline (YOLO11x + SAM + Multi-Pass)

### What Changed

| Area | Phase 1 | Phase 2 |
|------|---------|---------|
| Detection model | YOLOv8-small, 640px, 22.5 MB | YOLO11x, 1280px, 114 MB — mAP@50: 0.900 |
| Mask generation | `cv2.rectangle()` — rectangular | SAM ViT-B — pixel-precise contours |
| Inpainting | LaMa, single pass | LaMa, multi-pass with residual check |

### Model Downloads

```bash
# YOLO11x weights (114 MB)
uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('corzent/yolo11x_watermark_detection', 'best.pt', local_dir='models')
"

# SAM ViT-B checkpoint (357.7 MB) — from Meta CDN
uv run python -c "
import urllib.request, os; os.makedirs('models', exist_ok=True)
urllib.request.urlretrieve(
    'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
    'models/sam_vit_b_01ec64.pth'
)
"

# SAM Python package
uv add segment-anything
```

### Results vs Phase 1 (16 images, CPU)

| Metric | Phase 1 | Phase 2 |
|--------|---------|---------|
| Watermarks found | 11 | 20 (+82%) |
| Processing time | 17.4s | 125.9s |
| Mask type | Rectangular | Pixel-precise (SAM) |
| Residuals after pass 2 | N/A | 0 (all clean) |

> The higher detection count in Phase 2 is due to YOLO11x's improved recall (0.883) and 1280px input resolving small watermarks that YOLOv8-small at 640px missed.

---

## 6. Installation & Setup

### Prerequisites

- Python 3.10+
- [uv](https://astral.sh/uv) package manager

### Install uv (Windows)

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install Dependencies

```bash
cd "D:\pepagora\Watermark Remover"
uv sync
```

### Download Models (first-time only)

```bash
# YOLO11x detection model (114 MB)
uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('corzent/yolo11x_watermark_detection', 'best.pt', local_dir='models')
print('YOLO11x ready.')
"

# SAM ViT-B segmentation model (357.7 MB)
uv run python -c "
import urllib.request, os; os.makedirs('models', exist_ok=True)
urllib.request.urlretrieve(
    'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
    'models/sam_vit_b_01ec64.pth'
)
print('SAM ViT-B ready.')
"
```

> LaMa (`big-lama`) is downloaded automatically on first run and cached by PyTorch.

---

## 7. Usage

### Basic

```bash
uv run python watermark_remover.py -i "product sample images" -o "cleaned_images"
```

### With Debug Outputs

```bash
uv run python watermark_remover.py \
    -i "product sample images" \
    -o "cleaned_images" \
    --confidence 0.25 \
    --padding 10 \
    --max-passes 2 \
    --save-mask \
    --save-annotated
```

### Without SAM (faster, rectangular masks)

```bash
uv run python watermark_remover.py -i "input" -o "output" --no-sam
```

### GPU Mode

```bash
uv run python watermark_remover.py -i "input" -o "output" --device cuda
```

### Common Adjustments

| Goal | Command |
|------|---------|
| Fewer false positives | Add `-c 0.5` |
| Catch more watermarks | Add `-c 0.15` |
| Wider mask coverage | Add `-p 20` |
| Single pass (faster) | Add `--max-passes 1` |

### Output Structure

```
cleaned_images/
├── image1.jpg          ← cleaned image
├── image2.jpg
├── annotated/          ← original + detection boxes (--save-annotated)
│   └── *.jpg
└── masks/              ← binary inpainting masks (--save-mask)
    └── *.jpg
```

### Batch Summary (sample output)

```
====================================================
BATCH COMPLETE (YOLO11x + SAM + LaMa)
  Total images processed : 16
  Images with watermarks : 6
  Total watermarks found : 20
  Max passes per image   : 2
  Time elapsed           : 125.9s
  Output directory       : cleaned_images
====================================================
```

---

## 8. CLI Reference

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | required | Input directory containing images |
| `--output` | `-o` | required | Output directory for cleaned images |
| `--confidence` | `-c` | `0.25` | YOLO detection confidence threshold (0–1) |
| `--padding` | `-p` | `10` | Extra pixels around bbox for mask expansion |
| `--max-passes` | — | `2` | Maximum inpainting passes per image |
| `--no-sam` | — | `false` | Skip SAM; use rectangular bbox masks |
| `--save-mask` | — | `false` | Save binary masks to `output/masks/` |
| `--save-annotated` | — | `false` | Save annotated images to `output/annotated/` |
| `--device` | — | `cpu` | Inference device: `cpu` or `cuda` |

### Supported Image Formats

`.jpg` `.jpeg` `.png` `.bmp` `.webp` `.tiff`

---

## 9. Project Structure

```
Watermark Remover/
├── watermark_remover.py       # Main pipeline (Phase 2 — production)
├── watermark_detector.py      # Standalone detection script (dev/testing)
├── pyproject.toml             # Project dependencies (uv)
├── .python-version            # Python 3.10 pin
├── uv.lock                    # Locked dependency versions
├── .gitignore                 # Excludes models/, output dirs, sample images
│
├── models/                    # Model weights — git-ignored, download manually
│   ├── best.pt                # YOLO11x watermark detector (114 MB)
│   └── sam_vit_b_01ec64.pth   # SAM ViT-B segmentation model (357.7 MB)
│
├── DOCUMENTATION.md           # This file
└── RESEARCH_REPORT.md         # Alternatives evaluated + full upgrade analysis
```

---

## 10. Performance & Hardware

### Benchmarks

| Config | Images | Time | Avg/image |
|--------|--------|------|-----------|
| Phase 1 — CPU | 16 | 17.4s | 1.1s |
| Phase 2 — CPU | 16 | 125.9s | 7.9s |
| Phase 2 — GPU (estimated) | 16 | ~15–20s | ~1–1.5s |

> Phase 2 is slower on CPU because SAM ViT-B adds ~5s per image. On GPU this drops under 1s.

### Recommendations

| Hardware | Recommendation |
|----------|---------------|
| CPU | Use `--max-passes 1` and `--no-sam` for faster runs when quality is less critical |
| GPU (CUDA) | Use `--device cuda` — full Phase 2 pipeline runs in near real-time |
| Low memory | Reduce input image resolution before processing |

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| No watermarks detected | Confidence too high | Lower `-c` (e.g., `0.15`) |
| Too many false positives | Confidence too low | Raise `-c` (e.g., `0.5`) |
| Watermark edges still visible | Mask too tight | Increase `-p` (e.g., `20`) |
| Blurry inpainted area | Very large watermark | Expected for >30% coverage — LaMa limitation |
| SAM model not found | Checkpoint missing | Re-run SAM download from Section 6 |
| YOLO model not found | Weights missing | Re-run YOLO download from Section 6 |
| CUDA out of memory | Image too large for GPU | Fall back to `--device cpu` |

---

*Pepagora Watermark Remover — Last updated: Phase 2 (v2 pipeline).*


---

## 2. Phase 1 — Basic Pipeline (YOLOv8 + LaMa)

### Stack

| Component | Choice |
|-----------|--------|
| Detection | `mnemic/watermarks_yolov8` — YOLOv8-small (22.5 MB) |
| Masking | Rectangular bounding box + 10px padding |
| Inpainting | LaMa `big-lama` (196 MB) via `simple-lama-inpainting` |
| Package manager | `uv` |

### Pipeline

```
Input Image → YOLOv8 Detection → Rectangular Mask → LaMa Inpainting → Output
```

### Setup & Run

```bash
uv add ultralytics huggingface-hub simple-lama-inpainting
uv run python watermark_remover.py -i "product sample images" -o "cleaned_images"
```

### Results (16 images, CPU)

| Metric | Value |
|--------|-------|
| Images with watermarks | 7 |
| Watermarks found | 11 |
| Processing time | 17.4s (~1.1s/image) |

**Limitation:** Rectangular masks inpaint the bbox corners too (wasted area). YOLO11x's 640px input misses small watermarks.

---

## 3. Phase 2 — Upgraded Pipeline (YOLO11x + SAM + Multi-Pass)

### What Changed

| Area | Phase 1 | Phase 2 |
|------|---------|---------|
| Detection model | YOLOv8-small, 640px, 22.5 MB | YOLO11x, 1280px, 114 MB — mAP@50: 0.900 |
| Mask generation | `cv2.rectangle()` — rectangular | SAM ViT-B — pixel-precise contours |
| Inpainting | Single pass | Multi-pass: inpaint → re-detect → inpaint residuals |

### Pipeline

```
Input Image
    ↓
YOLO11x Detection (1280px input, mAP 0.900)
    ↓ bounding boxes
SAM ViT-B (pixel-precise mask from bbox prompt)
    ↓ binary mask
LaMa Inpainting — Pass 1
    ↓
Re-detect on result → if clean: done / else: SAM + LaMa again
    ↓
Output Image
```

### Model Downloads

```bash
# YOLO11x weights (114 MB)
uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('corzent/yolo11x_watermark_detection', 'best.pt', local_dir='models')
"

# SAM ViT-B checkpoint (357.7 MB) — from Meta CDN
uv run python -c "
import urllib.request, os; os.makedirs('models', exist_ok=True)
urllib.request.urlretrieve(
    'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
    'models/sam_vit_b_01ec64.pth'
)
"

# Install SAM dependency
uv add segment-anything
```

### Results (16 images, CPU)

| Metric | Phase 1 | Phase 2 |
|--------|---------|---------|
| Watermarks found | 11 | 20 (+82% more detections) |
| Processing time | 17.4s | 125.9s |
| Mask type | Rectangular | Pixel-precise |
| Residuals after pass 2 | N/A | 0 (all clean) |

---

## 4. Project Structure

```
Watermark Remover/
├── watermark_remover.py       # Main pipeline (Phase 2)
├── watermark_detector.py      # Standalone detection script
├── pyproject.toml             # Dependencies (uv)
├── models/                    # Model weights (git-ignored)
│   ├── best.pt                # YOLO11x (114 MB)
│   └── sam_vit_b_01ec64.pth   # SAM ViT-B (357.7 MB)
├── DOCUMENTATION.md
└── RESEARCH_REPORT.md         # Full alternatives & upgrade analysis
```

---

## 5. How to Run

### First-Time Setup

```bash
# 1. Install dependencies
uv sync

# 2. Download models (see Phase 2 section above)
```

### Run

```bash
# Basic
uv run python watermark_remover.py -i "product sample images" -o "cleaned_images"

# With debug outputs
uv run python watermark_remover.py -i "input" -o "output" --save-mask --save-annotated

# Without SAM (rectangular masks, faster)
uv run python watermark_remover.py -i "input" -o "output" --no-sam
```

---

## 6. CLI Reference

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | required | Input image directory |
| `--output` | `-o` | required | Output directory |
| `--confidence` | `-c` | 0.25 | Detection threshold (0–1) |
| `--padding` | `-p` | 10 | Pixels to expand mask around bbox |
| `--max-passes` | — | 2 | Max inpainting passes per image |
| `--no-sam` | — | false | Use rectangular masks instead of SAM |
| `--save-mask` | — | false | Save binary masks to `output/masks/` |
| `--save-annotated` | — | false | Save annotated images to `output/annotated/` |
| `--device` | — | cpu | `cpu` or `cuda` |

---

*Pepagora Watermark Remover — Last updated: Phase 2 (v2 pipeline).*