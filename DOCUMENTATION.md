# Automatic Watermark Remover

> A deep learning pipeline for automatically detecting and removing watermarks from product images.
>
> **Version:** 0.1.0 &nbsp;|&nbsp; **Python:** ≥ 3.10 &nbsp;|&nbsp; **Package Manager:** uv &nbsp;|&nbsp; **Branch:** phase-1

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Installation & Setup](#4-installation--setup)
5. [Usage](#5-usage)
6. [CLI Reference](#6-cli-reference)
7. [Project Structure](#7-project-structure)
8. [Performance](#8-performance)
9. [Troubleshooting](#9-troubleshooting)
10. [Limitations & Next Steps](#10-limitations--next-steps)

---

## 1. Overview

Watermarks are commonly embedded in product images to protect intellectual property. This project provides a fully automated pipeline that detects and removes visible watermarks at scale — with zero human intervention.

The system uses a **two-stage detect-then-inpaint** approach:

| Stage | Task | Model |
|-------|------|-------|
| **Stage 1** | Watermark Detection | YOLOv8-small (`mnemic/watermarks_yolov8`) |
| **Stage 2** | Watermark Removal | LaMa large-mask inpainting (`big-lama`) |

This decoupled architecture means each component can be independently tuned, swapped, or upgraded without touching the other.

---

## 2. Pipeline Architecture

### Stage Overview

```
Input Image
    │
    ▼
┌──────────────────────────────┐
│  Stage 1: Detection          │
│  YOLOv8-small (640px input)  │  → bounding boxes + confidence scores
└──────────────────┬───────────┘
                   │
                   ▼
┌──────────────────────────────┐
│  Stage 2: Mask Generation    │
│  cv2.rectangle() + padding   │  → binary mask (0 = keep, 255 = inpaint)
└──────────────────┬───────────┘
                   │
                   ▼
┌──────────────────────────────┐
│  Stage 3: Inpainting         │
│  LaMa big-lama               │  → watermark region reconstructed
└──────────────────┬───────────┘
                   │
                   ▼
┌──────────────────────────────┐
│  Stage 4: Output             │
│  Cleaned image               │  + optional masks/ + annotated/
└──────────────────────────────┘
```

### Stage Details

**Stage 1 — Watermark Detection**
YOLOv8-small scans the image and predicts bounding boxes around watermark regions. Each detection includes a bounding box `[x1, y1, x2, y2]`, a confidence score, and a class label. Only detections above `--confidence` threshold are kept.

**Stage 2 — Mask Generation**
Each detected bounding box is drawn onto a blank binary mask. A configurable `--padding` (default: 10px) expands each box outward to capture semi-transparent watermark edges.

**Stage 3 — Inpainting**
LaMa receives the original image and the binary mask. It reconstructs the masked region using surrounding context — filling the watermark area while preserving colors, textures, and patterns.

**Stage 4 — Output**
Saves the cleaned image. Optionally saves binary masks (`--save-mask`) and annotated detection images (`--save-annotated`). If no watermark is detected, the original image is copied unchanged.

### Model Loading

Both models are loaded **once** at batch start and reused across all images:

```
YOLOv8 (22.5 MB)  → downloaded from HuggingFace Hub → cached in ~/.cache/huggingface/hub/
LaMa   (196 MB)   → downloaded from GitHub Releases  → cached in torch hub dir
```

---

## 3. Technology Stack

### YOLOv8 — Watermark Detection

| Attribute | Details |
|-----------|---------|
| **Model** | `mnemic/watermarks_yolov8` (`watermarks_s_yolov8_v1.pt`) |
| **Architecture** | YOLOv8-small |
| **Input size** | 640 × 640 |
| **Size** | 22.5 MB |
| **Source** | HuggingFace Hub (auto-downloaded) |
| **Framework** | Ultralytics |

**Why YOLOv8?**
- Pre-trained specifically for watermark detection — no custom training needed
- Fast inference even on CPU, suitable for batch processing
- Provides confidence scores for threshold-based filtering
- Lightweight (22.5 MB) and loads quickly

### LaMa — Watermark Inpainting

| Attribute | Details |
|-----------|---------|
| **Model** | `big-lama` |
| **Architecture** | Fourier Convolution ResNet + Perceptual & Adversarial loss |
| **Size** | 196 MB |
| **Paper** | "Resolution-robust Large Mask Inpainting with Fourier Convolutions" (WACV 2022) |
| **Source** | GitHub Releases (auto-downloaded by `simple-lama-inpainting`) |

**Why LaMa?**
- Handles large masks — Fourier convolutions capture global image structure for coherent fills
- Resolution-robust — no fixed-size downscaling; works with varying product image dimensions
- Preserves textures (fabric, metal, gradients) better than traditional inpainting methods
- Clean pip-installable API: `result = lama(image, mask)`

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `ultralytics` | ≥ 8.4.19 | YOLOv8 inference engine |
| `huggingface-hub` | ≥ 1.5.0 | Model downloading and caching |
| `simple-lama-inpainting` | ≥ 0.1.2 | LaMa model wrapper |
| `opencv-python` | transitive | Mask drawing, annotation |
| `numpy` | transitive | Array / mask manipulation |
| `Pillow` | transitive | PIL image I/O |
| `torch` | transitive | PyTorch backend |
| `uv` | ≥ 0.7 | Package manager & virtual environment |

---

## 4. Installation & Setup

### Prerequisites

- Python 3.10+
- [uv](https://astral.sh/uv) package manager

### Install uv (Windows)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install Dependencies

```bash
cd "D:\pepagora\Watermark Remover"
uv sync
```

### First Run — Model Downloads

Both models are downloaded **automatically on first run** and cached — no manual download needed.

- **YOLOv8** (22.5 MB) → `~/.cache/huggingface/hub/`
- **LaMa** (196 MB) → torch hub cache directory

---

## 5. Usage

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
    --save-mask \
    --save-annotated
```

### GPU Mode

```bash
uv run watermark-remover -i "product sample images" -o "cleaned_images" --device cuda
```

### Alternative — run via Python directly

```bash
uv run python main.py -i "product sample images" -o "cleaned_images"
```

### Common Adjustments

| Goal | Flag |
|------|------|
| Fewer false positives | `-c 0.5` |
| Catch more watermarks | `-c 0.15` |
| Wider mask coverage | `-p 20` |

### Output Structure

```
cleaned_images/
├── image1.jpg          ← cleaned image (or original copy if no watermark)
├── image2.jpg
├── annotated/          ← original + red detection boxes  (--save-annotated)
│   └── *.jpg
└── masks/              ← binary inpainting masks         (--save-mask)
    └── *.jpg
```

### Batch Summary (sample output)

```
============================================================
BATCH COMPLETE
  Total images processed : 16
  Images with watermarks : 7
  Total watermarks found : 11
  Time elapsed           : 17.4s
  Output directory       : cleaned_images
============================================================
```

---

## 6. CLI Reference

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | required | Input directory containing images |
| `--output` | `-o` | required | Output directory for cleaned images |
| `--confidence` | `-c` | `0.25` | YOLO detection confidence threshold (0–1) |
| `--padding` | `-p` | `10` | Extra pixels around each bbox for mask expansion |
| `--save-mask` | — | `false` | Save binary masks to `output/masks/` |
| `--save-annotated` | — | `false` | Save annotated images to `output/annotated/` |
| `--device` | — | `cpu` | Inference device: `cpu` or `cuda` |

### Supported Image Formats

`.jpg` `.jpeg` `.png` `.bmp` `.webp` `.tiff`

---

## 7. Project Structure

```
Watermark Remover/
├── main.py                     ← CLI entry point
├── pyproject.toml              ← Dependencies & scripts (uv)
├── .python-version             ← Python 3.10 pin
├── uv.lock                     ← Locked dependency versions
├── .gitignore
│
├── watermark_remover/          ← Python package
│   ├── __init__.py             ← Public API
│   ├── config.py               ← Constants (model IDs, defaults)
│   ├── models.py               ← load_yolo_model(), load_lama_model()
│   ├── detector.py             ← detect_watermarks()
│   ├── masker.py               ← create_mask()
│   ├── inpainter.py            ← inpaint_image()
│   ├── pipeline.py             ← process_image(), process_batch()
│   └── cli.py                  ← argparse entry point
│
├── product sample images/      ← Input: raw product images with watermarks
│   └── *.jpg  (16 images)
│
├── cleaned_images/             ← Output: watermark-removed images
│   ├── *.jpg
│   ├── annotated/
│   └── masks/
│
├── DOCUMENTATION.md            ← This file
└── RESEARCH_REPORT.md          ← Alternatives evaluated & upgrade analysis
```

### Package Module Reference

| Module | Responsibility |
|--------|---------------|
| `config.py` | All constants — YOLO repo/filename, default confidence, padding |
| `models.py` | Model loaders — downloads YOLOv8 if missing, loads LaMa |
| `detector.py` | Runs YOLO inference, returns list of `{bbox, confidence, label}` dicts |
| `masker.py` | Converts detections to a binary uint8 numpy mask with padding |
| `inpainter.py` | Wraps LaMa — takes PIL image + mask, returns inpainted PIL image |
| `pipeline.py` | Orchestrates detect → mask → inpaint for single image and batch |
| `cli.py` | argparse CLI — parses args and calls `process_batch()` |

---

## 8. Performance

Tested on 16 real product images (mixed categories: electronics, textiles, food, machinery) on CPU.

| Metric | Value |
|--------|-------|
| Total images processed | 16 |
| Images with watermarks detected | 7 (43.75%) |
| Total watermark instances found | 11 |
| Total processing time | 17.4 seconds |
| Average time per image | ~1.1 seconds |
| Device | CPU |
| YOLOv8 model size | 22.5 MB |
| LaMa model size | 196 MB |
| Peak RAM usage | ~1.5 GB (estimated) |

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| No watermarks detected | Confidence threshold too high | Lower `-c` (e.g., `0.15`) |
| Too many false positives | Confidence threshold too low | Raise `-c` (e.g., `0.5`) |
| Watermark edges still visible | Mask too tight | Increase `-p` (e.g., `20–30`) |
| Blurry inpainted area | Very large watermark | Expected for >30% coverage — LaMa limitation |
| `ModuleNotFoundError` | venv not active | Run inside `uv run` |
| Model download fails | Network/HuggingFace issue | Retry; check internet connection |
| CUDA out of memory | Image too large for GPU | Switch to `--device cpu` |

---

## 10. Limitations & Next Steps

### Current Limitations

| Limitation | Description |
|------------|-------------|
| **Rectangular masks** | Masks are bboxes — inpaints corners even if only the watermark text is inside. Pixel-precise masks (SAM) would reduce this. |
| **Single-pass only** | No residual check after inpainting. A second detection pass could catch faint leftover marks. |
| **640px input** | YOLOv8-small processes at 640px — small watermarks on large images may be missed. |
| **Sequential processing** | Images are processed one at a time; no GPU batch parallelism. |

### Phase 2 Upgrade (Available on `phase-2` branch)

Phase 2 addresses all of the above:

| Improvement | Phase 2 Implementation |
|-------------|----------------------|
| Pixel-precise masks | SAM ViT-B — traces exact watermark boundary from bbox prompt |
| Multi-pass removal | Re-runs detection after inpainting; repeats until clean |
| Better detection | YOLO11x (1280px input, mAP@50: 0.900, 114 MB) |
| Results | 20 watermarks found vs 11 (+82%), 0 residuals after pass 2 |

---

*Pepagora Watermark Remover — Phase 1 (v0.1.0) | YOLOv8 + LaMa baseline pipeline.*
