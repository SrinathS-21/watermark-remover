# Automatic Watermark Remover

> A deep learning pipeline for automatically detecting and removing watermarks from product images.
>
> **Version:** 3.0.0 &nbsp;|&nbsp; **Python:** ≥ 3.10 &nbsp;|&nbsp; **Package Manager:** uv

---

## Table of Contents

1. [Overview](#1-overview)
2. [Key Features](#2-key-features)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Phase 1 — Basic Pipeline (YOLOv8 + LaMa)](#4-phase-1--basic-pipeline-yolov8--lama)
5. [Phase 2 — Upgraded Pipeline (YOLOv8 + SAM + Multi-Pass)](#5-phase-2--upgraded-pipeline-yolov8--sam--multi-pass)
6. [Phase 3 — Dual-Model Pipeline (YOLO11x + YOLOv8 + SAM + LaMa)](#6-phase-3--dual-model-pipeline-yolo11x--yolov8--sam--lama)
7. [Installation & Setup](#7-installation--setup)
8. [Usage](#8-usage)
9. [CLI Reference](#9-cli-reference)
10. [Project Structure](#10-project-structure)
11. [Performance & Hardware](#11-performance--hardware)

---

## 1. Overview

Watermarks are commonly embedded in images to protect intellectual property. In legitimate workflows — such as dataset cleaning, image restoration, or content editing — there is a need to automatically detect and remove them at scale.

This project provides a fully automated, end-to-end pipeline that:

- Detects watermarks using a trained object detection model
- Generates precise masks for the detected regions
- Removes watermarks using a high-quality deep inpainting model
- Saves cleaned images with optional debug outputs (masks, annotated detections)

The pipeline was built in three phases. Phase 1 established a working baseline; Phase 2 upgraded masking and added multi-pass; Phase 3 introduces a dual-model strategy for best-in-class recall and precision.

---

## 2. Key Features

| Feature | Description |
|---------|-------------|
| **Dual-Model Detection** | YOLO11x (high recall) sweeps pass 1; YOLOv8 (tight boxes) checks residuals in pass 2 |
| **Pixel-Precise Masks** | SAM (Segment Anything Model) traces the exact watermark boundary instead of using rectangles |
| **High-Quality Inpainting** | LaMa (Large Mask Inpainting) reconstructs the background using surrounding context |
| **Two-Pass Removal** | Pass 1 catches all watermarks; pass 2 finds any residuals on the now-clean background |
| **Batch Processing** | Processes entire folders automatically |
| **Hardware Flexible** | Runs on CPU or GPU (`--device cuda`) |
| **Debug Outputs** | Optional binary masks and annotated detection images |

---

## 3. Pipeline Architecture

The system follows a dual-model two-pass deep learning pipeline.

### Stage Overview

```
Input Image
    │
    ▼
┌──────────────────────────────┐
│  Pass 1 — Stage 1: Detection │
│  YOLO11x (high recall)       │  → bounding boxes + confidence scores
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Pass 1 — Stage 2: Masking   │
│  SAM ViT-B (pixel-precise)   │  → binary mask tracing watermark boundary
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Pass 1 — Stage 3: Inpaint   │
│  LaMa big-lama               │  → watermark region reconstructed
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Pass 2 — Stage 4: Detection │
│  YOLOv8 (precision, tight)   │  → residual check on clean background
└──────────────┬───────────────┘
               │
         residuals?
          yes │  no ──────────────┐
               │                  │
               ▼                  ▼
┌──────────────────────────────┐  │
│  Pass 2 — Stage 5: Masking   │  │
│  SAM ViT-B                   │  │
└──────────────┬───────────────┘  │
               │                  │
               ▼                  │
┌──────────────────────────────┐  │
│  Pass 2 — Stage 6: Inpaint   │  │
│  LaMa big-lama               │  │
└──────────────┬───────────────┘  │
               │                  │
               └──────────────────┘
                        │
                        ▼
               ┌────────────────┐
               │  Stage 7: Out  │  → cleaned image + masks/ + annotated/
               └────────────────┘
```

### Why Two Models?

| Model | Role | Strength |
|-------|------|----------|
| **YOLO11x** (114 MB, 1280px) | Pass 1 — initial sweep | High recall — catches every watermark even at low confidence |
| **YOLOv8** (22.5 MB, 640px) | Pass 2 — residual check | Tight bounding boxes — clean prompt for SAM on plain background |

YOLO11x is used first because it has the highest recall — it catches all watermarks on the original complex background, even if its bounding boxes are sometimes wide or fragmented. SAM and LaMa handle those detections correctly regardless of box tightness.

After pass 1, the background is plain and uniform. YOLOv8 then runs on this clean result: its tight boxes give SAM a precise, unambiguous prompt with no product texture around the watermark — eliminating the SAM bleeding problem that occurred on the original image.

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

## 5. Phase 2 — Upgraded Pipeline (YOLOv8 + SAM + Multi-Pass)

### What Changed

| Area | Phase 1 | Phase 2 |
|------|---------|---------|
| Detection model | YOLOv8-small (same model, now with SAM) | YOLOv8-small, 22.5 MB — retained after YOLO11x evaluation |
| Mask generation | `cv2.rectangle()` — rectangular | SAM ViT-B — pixel-precise contours |
| Inpainting | LaMa, single pass | LaMa, multi-pass with residual check |

### Why YOLOv8 over YOLO11x

During development, we evaluated YOLO11x (`corzent/yolo11x_watermark_detection`, 114 MB, 1280px input, mAP@50: 0.900) as a potential upgrade. However, testing on our dataset revealed a critical issue:

- **YOLO11x fragments single watermarks into multiple overlapping bounding boxes.** A single watermark that YOLOv8 correctly detects as one box gets split into 2–3 overlapping boxes by YOLO11x. This fragmentation creates overlapping SAM masks and degrades inpainting quality.
- **YOLOv8 detects more unique watermarks.** In a side-by-side comparison with IoU-based box merging, YOLOv8 detected **9 unique watermarks across 7 images** vs YOLO11x's **8 across 6 images**.
- **Simpler and more reliable.** YOLOv8's clean one-box-per-watermark behavior works better with the downstream SAM + LaMa pipeline.

**Conclusion:** Higher mAP on a benchmark does not always mean better real-world performance. YOLOv8's detections are cleaner and more reliable for this pipeline.

### Model Downloads

```bash
# YOLOv8 weights (22.5 MB) — auto-downloaded by pipeline, or manually:
uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('mnemic/watermarks_yolov8', 'watermarks_s_yolov8_v1.pt', local_dir='models')
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
| Images with watermarks | 7 | 7 |
| Total watermarks found | 11 | 11 |
| Processing time | 17.4s | 68.9s |
| Mask type | Rectangular | Pixel-precise (SAM) |
| Residuals after pass 2 | N/A | 0 (all clean) |

> Phase 2 detects the same watermarks as Phase 1 but removes them far more cleanly — SAM traces the exact watermark boundary and multi-pass ensures no residuals remain.

---

## 6. Phase 3 — Dual-Model Pipeline (YOLO11x + YOLOv8 + SAM + LaMa)

### What Changed

| Area | Phase 2 | Phase 3 |
|------|---------|---------|
| Pass 1 detection | YOLOv8-small (22.5 MB) | YOLO11x (114 MB, 1280px) — higher recall |
| Pass 2 detection | YOLOv8 again | YOLOv8 — tight bbox on clean background |
| SAM in pass 2 | Prompted on original image | Prompted on post-inpaint clean result |
| SAM bleeding | Can occur on complex original | Eliminated — plain background in pass 2 |

### Motivation

Phase 2 used YOLOv8 for both passes. While effective, it had two limitations:

1. **YOLOv8 may miss low-confidence watermarks on the original complex background.** YOLO11x, with its 1280px input and deeper architecture, has higher recall on difficult cases.
2. **SAM bleeding on pass 2.** When YOLOv8's bounding box is wide on the original image, SAM sometimes segments product features instead of the watermark (since the box overlaps product texture).

Phase 3 addresses both: YOLO11x sweeps the original image with maximum recall, and after LaMa cleans it, YOLOv8 runs on the inpainted result where the background is plain — giving SAM a clean, unambiguous prompt.

### Model Downloads

```bash
# YOLO11x weights (114 MB) — auto-downloaded by pipeline, or manually:
uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('corzent/yolo11x_watermark_detection', 'best.pt', local_dir='models')
print('YOLO11x ready.')
"

# YOLOv8 weights (22.5 MB) — auto-downloaded by pipeline, or manually:
uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('mnemic/watermarks_yolov8', 'watermarks_s_yolov8_v1.pt', local_dir='models')
print('YOLOv8 ready.')
"

# SAM ViT-B checkpoint (357.7 MB)
uv run python -c "
import urllib.request, os; os.makedirs('models', exist_ok=True)
urllib.request.urlretrieve(
    'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
    'models/sam_vit_b_01ec64.pth'
)
print('SAM ViT-B ready.')
"
```

### Results vs Phase 2 (16 images, CPU)

| Metric | Phase 2 | Phase 3 |
|--------|---------|---------|
| Images with watermarks | 7 | 6 |
| Total watermarks found | 11 | 21 |
| Pass 1 detections | 11 (YOLOv8) | 20 (YOLO11x) |
| Pass 2 residuals fixed | 0 | 1 (BitterGourd) |
| Processing time | 68.9s | 119.0s |
| Mask type | Pixel-precise (SAM) | Pixel-precise (SAM) |

> YOLO11x's higher recall in pass 1 found significantly more watermark instances (20 vs 11). The one BitterGourd residual caught by YOLOv8 in pass 2 demonstrates the dual-model strategy working as designed — YOLO11x cleaned the main region, and YOLOv8 caught the surviving trace on the clean background.

---

## 7. Installation & Setup

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
# YOLO11x detection model — pass 1 (114 MB)
uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('corzent/yolo11x_watermark_detection', 'best.pt', local_dir='models')
print('YOLO11x ready.')
"

# YOLOv8 detection model — pass 2 (22.5 MB)
uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('mnemic/watermarks_yolov8', 'watermarks_s_yolov8_v1.pt', local_dir='models')
print('YOLOv8 ready.')
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
> Both YOLO models are also auto-downloaded on first run if the weight files are missing.

---

## 7. Usage

### Basic

```bash
uv run watermark-remover -i "product sample images" -o "cleaned_images"
```

### With Debug Outputs

```bash
uv run watermark-remover \
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
uv run watermark-remover -i "input" -o "output" --no-sam
```

### GPU Mode

```bash
uv run watermark-remover -i "input" -o "output" --device cuda
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
============================================================
BATCH COMPLETE (YOLO11x + YOLOv8 + SAM + LaMa)
  Total images processed : 16
  Images with watermarks : 6
  Total watermarks found : 21
  Max passes per image   : 2
  Time elapsed           : 119.0s
  Output directory       : cleaned_images_v3
============================================================
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
├── main.py                        # Entry point — delegates to watermark_remover.cli
├── pyproject.toml                 # Project dependencies (uv) + CLI script entry
├── .python-version                # Python 3.10 pin
├── uv.lock                        # Locked dependency versions
├── .gitignore                     # Excludes models/, output dirs, sample images
│
├── watermark_remover/             # Core package
│   ├── __init__.py                # Public API exports
│   ├── config.py                  # Constants, thresholds, model paths (both YOLO models)
│   ├── models.py                  # YOLO11x, YOLOv8, SAM, LaMa model loaders
│   ├── detector.py                # YOLO inference → detection dicts
│   ├── masker.py                  # SAM / bbox mask generation
│   ├── inpainter.py               # LaMa inpainting wrapper
│   ├── pipeline.py                # Orchestration: dual-model two-pass detect + inpaint
│   └── cli.py                     # argparse CLI with all flags
│
├── models/                        # Model weights — git-ignored, download manually
│   ├── best.pt                    # YOLO11x watermark detector (114 MB) — pass 1
│   ├── watermarks_s_yolov8_v1.pt  # YOLOv8 watermark detector (22.5 MB) — pass 2
│   └── sam_vit_b_01ec64.pth       # SAM ViT-B segmentation model (357.7 MB)
│
├── DOCUMENTATION.md               # This file
└── RESEARCH_REPORT.md             # Alternatives evaluated + full upgrade analysis
```

---

## 10. Performance & Hardware

### Benchmarks

| Config | Images | Time | Avg/image |
|--------|--------|------|-----------|
| Phase 1 — CPU | 16 | 17.4s | 1.1s |
| Phase 2 — CPU | 16 | 68.9s | 4.3s |
| Phase 3 — CPU | 16 | 119.0s | 7.4s |
| Phase 3 — GPU (estimated) | 16 | ~15–20s | ~1–1.3s |

> Phase 3 adds YOLO11x on top of Phase 2, adding ~50s on CPU (two model loads + larger 1280px inference). On GPU both detection passes drop under 1s each.

### Recommendations

| Hardware | Recommendation |
|----------|---------------|
| CPU | Use `--max-passes 1` and `--no-sam` for faster runs when quality is less critical |
| GPU (CUDA) | Use `--device cuda` — full Phase 3 pipeline runs in near real-time |
| Low memory | Reduce input image resolution before processing |

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| No watermarks detected | Confidence too high | Lower `-c` (e.g., `0.15`) |
| Too many false positives | Confidence too low | Raise `-c` (e.g., `0.5`) |
| Watermark edges still visible | Mask too tight | Increase `-p` (e.g., `20`) |
| Blurry inpainted area | Very large watermark | Expected for >30% coverage — LaMa limitation |
| SAM model not found | Checkpoint missing | Re-run SAM download from Section 7 |
| YOLO model not found | Weights missing | Re-run YOLO download from Section 7 |
| CUDA out of memory | Image too large for GPU | Fall back to `--device cpu` |

---

*Pepagora Watermark Remover — Last updated: Phase 3 (v3.0.0).*