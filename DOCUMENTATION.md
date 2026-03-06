# Watermark Remover — Technical Documentation

> **Automatic Visible Watermark Removal System for Product Images**
>
> Version: 0.1.0 | Python ≥ 3.10 | License: Internal

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Approach](#2-system-approach)
3. [Architecture](#3-architecture)
4. [Technology Stack — What We Have & Why](#4-technology-stack--what-we-have--why)
5. [Rejected Alternatives & Rationale](#5-rejected-alternatives--rationale)
6. [Differentiation from Alternative Solutions](#6-differentiation-from-alternative-solutions)
7. [Project Structure](#7-project-structure)
8. [Standard Operating Procedure (SOP)](#8-standard-operating-procedure-sop)
9. [Configuration Reference](#9-configuration-reference)
10. [Performance Benchmarks](#10-performance-benchmarks)
11. [Limitations & Future Roadmap](#11-limitations--future-roadmap)

---

## 1. Executive Summary

The **Watermark Remover** is an AI-powered pipeline designed to automatically detect and remove visible watermarks (text overlays, logos, semi-transparent stamps) from product images at scale.

The system uses a **two-stage approach**:

| Stage | Task | Model |
|-------|------|-------|
| **Stage 1** | Watermark Detection | YOLOv8 (mnemic/watermarks_yolov8) |
| **Stage 2** | Watermark Inpainting | LaMa — Large Mask Inpainting (big-lama) |

This decoupled architecture ensures each component can be independently upgraded, tuned, or replaced without affecting the other.

---

## 2. System Approach

### 2.1 Problem Statement

Product catalog images from various sources often contain visible watermarks — company logos, text overlays, or semi-transparent stamps. These watermarks degrade the visual quality and usability of product imagery. Manual removal is time-consuming and does not scale.

### 2.2 Design Philosophy

We follow a **Detect → Mask → Inpaint** three-step approach:

1. **Detect**: Locate watermark regions in the image using a trained object detection model.
2. **Mask**: Convert detected bounding boxes into a binary mask (black = keep, white = remove) with configurable padding to capture watermark edges.
3. **Inpaint**: Feed the original image and the mask to a deep inpainting model that reconstructs the underlying content naturally.

### 2.3 Why This Approach?

| Design Decision | Rationale |
|----------------|-----------|
| **Separate detection and removal** | Unlike end-to-end models that attempt both simultaneously (e.g., FODUU), separating the stages gives precise control — we know *what* the model removes and *where*. |
| **Object detection for localization** | YOLO provides pixel-precise bounding boxes with confidence scores, enabling threshold-based filtering and visual debugging. |
| **Inpainting-based removal** | Inpainting fills masked regions using learned priors of natural image structure, producing seamless results even for complex textures. The alternative (direct pixel-to-pixel translation) often blurs or hallucinates unwanted artifacts. |
| **Mask padding** | Watermark edges are often semi-transparent or blended. Adding configurable padding (default: 10px) around detected bounding boxes ensures complete removal without visible ghosting. |

---

## 3. Architecture

### 3.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    WATERMARK REMOVER PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐     ┌───────────────┐     ┌──────────────┐      │
│   │  Input    │     │   YOLOv8      │     │  Detection   │      │
│   │  Image    │────▶│  Detector     │────▶│  Results     │      │
│   │  (RGB)   │     │  (mnemic/     │     │  [bbox, conf,│      │
│   │          │     │  watermarks_  │     │   label]     │      │
│   └──────────┘     │  yolov8)      │     └──────┬───────┘      │
│                    └───────────────┘            │               │
│                                                 │               │
│                                                 ▼               │
│                                    ┌────────────────────┐       │
│                                    │   Mask Generator   │       │
│                                    │   ─────────────    │       │
│                                    │  • Convert bbox    │       │
│                                    │    to binary mask  │       │
│                                    │  • Apply padding   │       │
│                                    │  • 0=keep 255=fill │       │
│                                    └─────────┬──────────┘       │
│                                              │                  │
│   ┌──────────┐     ┌───────────────┐         │                  │
│   │  Output   │     │   LaMa        │         │                  │
│   │  Clean    │◀────│  Inpainter    │◀────────┘                  │
│   │  Image   │     │  (big-lama)   │   Image + Mask             │
│   │          │     │               │                            │
│   └──────────┘     └───────────────┘                            │
│                                                                 │
│   Optional Outputs:                                             │
│   ├── annotated/   (images with detection bboxes drawn)         │
│   └── masks/       (binary masks for debugging/review)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow Diagram

```
                    ┌─────────────────────┐
                    │   Batch of Images    │
                    │   (JPG/PNG/WebP/BMP) │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Image Iterator    │
                    │   (sorted, filtered │
                    │    by extension)    │
                    └──────────┬──────────┘
                               │
              ┌────────────────▼────────────────┐
              │         Per-Image Pipeline       │
              │                                  │
              │  ┌──────────────────────────┐    │
              │  │ 1. YOLO Inference        │    │
              │  │    conf ≥ threshold       │    │
              │  └────────────┬─────────────┘    │
              │               │                  │
              │          Detections?             │
              │          /        \              │
              │        Yes         No            │
              │         │           │            │
              │  ┌──────▼──────┐   │            │
              │  │ 2. Create   │   │            │
              │  │    Mask     │   │            │
              │  └──────┬──────┘   │            │
              │         │          │            │
              │  ┌──────▼──────┐   │            │
              │  │ 3. LaMa     │   │            │
              │  │  Inpainting │   │            │
              │  └──────┬──────┘   │            │
              │         │          │            │
              │  ┌──────▼──────────▼─────┐      │
              │  │ 4. Save Output        │      │
              │  │    (+ mask, annotated) │      │
              │  └───────────────────────┘      │
              └────────────────┬────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Batch Summary     │
                    │   (stats + logging) │
                    └─────────────────────┘
```

### 3.3 Model Loading & Caching

```
┌──────────────────────────────────────────────────────────┐
│                    MODEL MANAGEMENT                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  YOLOv8 Model (22.5 MB)                                 │
│  ├── Source: HuggingFace hub (mnemic/watermarks_yolov8)  │
│  ├── File: watermarks_s_yolov8_v1.pt                     │
│  ├── Cache: ~/.cache/huggingface/hub/                    │
│  └── Load: ultralytics.YOLO(path)                        │
│                                                          │
│  LaMa Model (196 MB)                                    │
│  ├── Source: GitHub Releases (auto-download)             │
│  ├── File: big-lama.pt                                   │
│  ├── Cache: torch hub checkpoints dir                    │
│  └── Load: SimpleLama()                                  │
│                                                          │
│  Both models are loaded ONCE at batch start and reused   │
│  across all images for efficiency.                       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 4. Technology Stack — What We Have & Why

### 4.1 YOLOv8 — Watermark Detection

| Attribute | Details |
|-----------|---------|
| **Model** | `mnemic/watermarks_yolov8` (`watermarks_s_yolov8_v1.pt`) |
| **Source** | HuggingFace Hub |
| **Architecture** | YOLOv8-small |
| **Size** | 22.5 MB |
| **Classes** | `{0: 'watermark', 1: 'watermark'}` |
| **Framework** | Ultralytics |

**Why YOLOv8?**
- **Real-time performance**: YOLOv8 is one of the fastest object detection architectures, making it suitable for batch processing hundreds of images.
- **Pre-trained on watermarks**: The `mnemic/watermarks_yolov8` model is specifically fine-tuned for watermark detection, eliminating the need for custom training.
- **Bounding box output**: Provides precise spatial coordinates of watermarks, which directly translate to inpainting masks.
- **Confidence scoring**: Each detection includes a confidence score, allowing threshold-based filtering to balance precision vs. recall.
- **Lightweight**: At only 22.5 MB, the model loads quickly and runs efficiently even on CPU.

### 4.2 LaMa (Large Mask Inpainting) — Watermark Removal

| Attribute | Details |
|-----------|---------|
| **Model** | `big-lama` |
| **Source** | GitHub Releases (auto-downloaded) |
| **Architecture** | Fourier Convolution-based ResNet with Perceptual + Adversarial loss |
| **Size** | 196 MB |
| **Paper** | "Resolution-robust Large Mask Inpainting with Fourier Convolutions" (WACV 2022) |
| **Wrapper** | `simple-lama-inpainting` (pip package) |

**Why LaMa?**
- **Handles large masks**: Unlike traditional inpainting methods that struggle with large missing regions, LaMa uses Fourier convolutions to capture global image structure, producing coherent fills even for large watermark areas.
- **Resolution-robust**: Works well across different image resolutions without requiring resizing to a fixed dimension. Product images vary widely in resolution — LaMa handles this gracefully.
- **Texture continuity**: The Perceptual + Adversarial training regime ensures the filled region matches surrounding textures, critical for product images with repeating patterns (fabric, metal, etc.).
- **Battle-tested**: LaMa is one of the most widely-adopted inpainting models in production, with extensive benchmarks on Places2 and other datasets.
- **Simple integration**: The `simple-lama-inpainting` package provides a clean API — `result = lama(image, mask)` — with zero configuration needed.

### 4.3 Supporting Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `ultralytics` | ≥ 8.4.19 | YOLOv8 inference engine |
| `huggingface-hub` | ≥ 1.5.0 | Model downloading and caching |
| `simple-lama-inpainting` | ≥ 0.1.2 | LaMa model wrapper |
| `opencv-python` | (transitive) | Image I/O, mask drawing, annotation |
| `numpy` | (transitive) | Array/mask manipulation |
| `Pillow` | (transitive) | PIL image handling |
| `torch` | (transitive) | PyTorch backend for both models |

### 4.4 Package Manager

| Tool | Why |
|------|-----|
| **uv** (v0.7+) | Ultra-fast Python package manager; handles virtual environments, dependency resolution, and Python version management. Chosen over pip/conda for speed and reproducibility. |

---

## 5. Rejected Alternatives & Rationale

### 5.1 IOPaint (formerly LaMa Cleaner)

| Attribute | Details |
|-----------|---------|
| **What it is** | GUI/CLI tool for interactive inpainting (used to wrap LaMa, MAT, etc.) |
| **Why rejected** | **Archived** (August 2025). Import chain is broken — `diffusers==0.27.2` uses `cached_download` which was removed from `huggingface_hub>=1.0`. Installation succeeds but runtime fails with `ImportError`. No fixes will be released. |
| **Impact** | We use `simple-lama-inpainting` instead, which wraps the same LaMa model with a simple, dependency-light API that remains compatible with current library versions. |

### 5.2 FODUU WatermarkRemover (foduucom/Watermark_Removal)

| Attribute | Details |
|-----------|---------|
| **What it is** | End-to-end U-Net encoder-decoder model (126 MB) that directly outputs watermark-free images. |
| **Why rejected** | **Poor output quality**. The model processes the entire image at a 256×256 bottleneck resolution, meaning: (1) It blurs/alters non-watermarked areas, (2) it loses fine detail in product images, (3) it introduces color shifts. Tested on all 16 sample images — user verdict: "too worst". |
| **Key lesson** | End-to-end models that process the entire image suffer when watermarks are small relative to the image. Our detect-then-inpaint approach only modifies the watermark region, preserving the rest of the image perfectly. |

### 5.3 noai-watermark

| Attribute | Details |
|-----------|---------|
| **What it is** | Tool for removing invisible AI watermarks (like SynthID, Tree-Ring, etc.) |
| **Why rejected** | **Wrong problem domain**. It targets steganographic/invisible watermarks embedded in AI-generated images. Our use case is removing *visible* text/logo watermarks from product photos. This tool has zero applicability to our scenario. |

### 5.4 zuruoke/watermark-removal

| Attribute | Details |
|-----------|---------|
| **What it is** | TensorFlow-based watermark removal (neural network approach) |
| **Why rejected** | **Dead project**. Requires TensorFlow 1.15 (Python 3.7 era, released 2019). The project's own Google Colab notebook is marked as "broken". TF 1.x is EOL, incompatible with modern Python, and would require a completely separate runtime environment. |

### 5.5 Summary Comparison Table

| Solution | Detection | Removal | Status | Quality | Why Not |
|----------|-----------|---------|--------|---------|---------|
| **Our System (YOLO+LaMa)** | ✅ YOLOv8 | ✅ LaMa | Active | ★★★★☆ | — *Selected* |
| IOPaint | ❌ Manual | ✅ LaMa/MAT | ❌ Archived | N/A | Broken imports |
| FODUU | ❌ End-to-end | ❌ End-to-end | Active | ★☆☆☆☆ | Blurs entire image |
| noai-watermark | N/A | ❌ Invisible only | Active | N/A | Wrong domain |
| zuruoke | ❌ End-to-end | ❌ End-to-end | ❌ Dead | N/A | TF 1.15 required |

---

## 6. Differentiation from Alternative Solutions

### 6.1 Vs. End-to-End Models (FODUU, zuruoke)

| Aspect | End-to-End | Our System |
|--------|-----------|------------|
| **Scope of modification** | Processes entire image through encoder-decoder | Only modifies detected watermark regions |
| **Detail preservation** | Bottleneck degrades resolution (e.g., 256×256) | Original pixels preserved outside mask |
| **Transparency** | Black-box — no visibility into what was modified | Detection bboxes and masks available for review |
| **Confidence filtering** | None — always processes | Confidence threshold filters false positives |
| **Multi-watermark handling** | Model must learn to handle 0-N watermarks | Each detection handled independently |

### 6.2 Vs. Manual Inpainting Tools (IOPaint, Photoshop)

| Aspect | Manual Tools | Our System |
|--------|-------------|------------|
| **Automation** | Requires human to draw mask for each image | Fully automatic — zero human intervention |
| **Scalability** | 1-5 images/hour (manual masking) | 16 images in 17.4s (batch automated) |
| **Consistency** | Varies by operator skill | Deterministic results for same parameters |
| **Cost** | Requires trained operator | One-time setup, runs unattended |

### 6.3 Vs. Academic Research Models (SLBR, deep-blind-watermark-removal)

| Aspect | Academic Models | Our System |
|--------|----------------|------------|
| **Deployment readiness** | Research code, custom training required | Production-ready CLI with pip-installable deps |
| **Pre-trained availability** | Often limited to paper's dataset | Pre-trained on diverse watermark styles |
| **Dependency weight** | Custom frameworks, CUDA-specific builds | Standard PyTorch ecosystem |
| **Maintenance** | Papers published, code rarely maintained | Active upstream libraries (ultralytics, torch) |

### 6.4 Vs. Traditional Image Processing (OpenCV inpainting)

| Aspect | OpenCV (`cv2.inpaint`) | Our System (LaMa) |
|--------|----------------------|-------------------|
| **Algorithm** | Navier-Stokes / Telea (pixel diffusion) | Deep learning (Fourier convolutions + adversarial training) |
| **Large masks** | Blurry, smeared results | Coherent texture synthesis |
| **Texture understanding** | None — purely local interpolation | Learned global context from millions of images |
| **Complex patterns** | Fails on fabric, text backgrounds | Handles textures, gradients, structured backgrounds |

---

## 7. Project Structure

```
Watermark Remover/
├── watermark_remover.py      # Main pipeline — detect + mask + inpaint (production)
├── watermark_detector.py     # Standalone YOLO detection script (development/testing)
├── main.py                   # Placeholder entry point
├── pyproject.toml            # Project config & dependencies
├── .python-version           # Python version pin (3.10)
├── uv.lock                   # Dependency lock file
├── .gitignore                # Git ignore rules
│
├── product sample images/    # Input: raw product images with watermarks
│   ├── FreshApple1663381094_9190.jpg
│   ├── SilkSarees1042186230_5783.jpg
│   └── ... (16 images)
│
├── cleaned_images/           # Output: watermark-removed images
│   ├── FreshApple1663381094_9190.jpg    (clean)
│   ├── annotated/                       (bboxes drawn on original)
│   │   └── *.jpg
│   └── masks/                           (binary inpainting masks)
│       └── *.jpg
│
└── detected_watermarks/      # Output from standalone detector
    └── annotated/
        └── *.jpg
```

### Key Files

| File | Role | Entry Point |
|------|------|-------------|
| `watermark_remover.py` | Full production pipeline (detect → mask → inpaint) | `python watermark_remover.py --input <dir> --output <dir>` |
| `watermark_detector.py` | Detection-only script for testing/debugging | `python watermark_detector.py` |
| `pyproject.toml` | Dependency specification for `uv` | `uv sync` |

---

## 8. Standard Operating Procedure (SOP)

### 8.1 Environment Setup (One-Time)

```bash
# Step 1: Install uv (if not already installed)
# Windows (PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Step 2: Navigate to project directory
cd "D:\pepagora\Watermark Remover"

# Step 3: Pin Python version and create virtual environment
uv python pin 3.10
uv venv --python 3.10

# Step 4: Install all dependencies
uv sync

# Step 5: Activate virtual environment
.venv\Scripts\activate
```

### 8.2 Running the Pipeline

#### Basic Usage

```bash
python watermark_remover.py \
    --input "product sample images" \
    --output "cleaned_images"
```

#### Full Usage with All Options

```bash
python watermark_remover.py \
    --input "product sample images" \
    --output "cleaned_images" \
    --confidence 0.25 \
    --padding 10 \
    --save-mask \
    --save-annotated \
    --device cpu
```

#### Quick Command Reference

| Action | Command |
|--------|---------|
| Basic cleanup | `python watermark_remover.py -i <input_dir> -o <output_dir>` |
| Higher precision (fewer false positives) | `python watermark_remover.py -i <input> -o <output> -c 0.5` |
| Higher recall (catch more watermarks) | `python watermark_remover.py -i <input> -o <output> -c 0.15` |
| Wider mask coverage | `python watermark_remover.py -i <input> -o <output> -p 20` |
| Save debug outputs | `python watermark_remover.py -i <input> -o <output> --save-mask --save-annotated` |
| GPU acceleration | `python watermark_remover.py -i <input> -o <output> --device cuda` |

### 8.3 First Run Behavior

On the **first run**, the system will automatically download:

1. **YOLOv8 model** (22.5 MB) from HuggingFace Hub → cached in `~/.cache/huggingface/hub/`
2. **LaMa model** (196 MB) from GitHub Releases → cached in torch hub checkpoints directory

Subsequent runs reuse the cached models with zero additional download.

### 8.4 Interpreting Results

After batch processing, the system logs a summary:

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

**Output directory structure:**

| Path | Content |
|------|---------|
| `<output>/` | Cleaned images (watermarks removed) or copies of originals (if no watermark detected) |
| `<output>/masks/` | Binary masks showing detected regions (white = inpainted, black = preserved). Only generated with `--save-mask`. |
| `<output>/annotated/` | Original images with red bounding boxes drawn around detected watermarks. Only generated with `--save-annotated`. |

### 8.5 Quality Review Process

1. **Run with debug outputs**: Add `--save-mask --save-annotated` flags.
2. **Check annotated images**: Verify detections are correct (no false positives on product text/logos).
3. **Check masks**: Ensure mask coverage is sufficient (increase `--padding` if watermark edges remain visible).
4. **Check cleaned images**: Verify inpainting looks natural.
5. **Adjust confidence**: If false positives → increase `--confidence`; if missed watermarks → decrease it.
6. **Re-run** with adjusted parameters on the same batch.

### 8.6 Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| No watermarks detected | Confidence threshold too high | Lower `--confidence` (e.g., 0.15) |
| False positive detections | Confidence threshold too low | Raise `--confidence` (e.g., 0.5) |
| Watermark ghost/edges visible | Mask too tight | Increase `--padding` (e.g., 20-30) |
| Inpainted area looks blurry | Large watermark covering critical area | Expected for very large masks; LaMa limitation |
| `ModuleNotFoundError` | Virtual environment not activated | Run `.venv\Scripts\activate` |
| Model download fails | Network issue or HuggingFace outage | Retry; check internet connection |
| CUDA out of memory | Image too large for GPU | Use `--device cpu` or reduce image resolution before processing |

---

## 9. Configuration Reference

### 9.1 CLI Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--input` | `-i` | str | *required* | Input directory containing images |
| `--output` | `-o` | str | *required* | Output directory for cleaned images |
| `--confidence` | `-c` | float | 0.25 | YOLO detection confidence threshold (0.0–1.0) |
| `--padding` | `-p` | int | 10 | Extra pixels around each detection bbox for mask expansion |
| `--save-mask` | — | flag | false | Save binary masks to `<output>/masks/` |
| `--save-annotated` | — | flag | false | Save annotated images to `<output>/annotated/` |
| `--device` | — | str | cpu | Inference device: `cpu` or `cuda` |

### 9.2 Supported Image Formats

`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`

### 9.3 Constants (in source)

| Constant | Value | Description |
|----------|-------|-------------|
| `YOLO_REPO` | `mnemic/watermarks_yolov8` | HuggingFace model repository |
| `YOLO_FILENAME` | `watermarks_s_yolov8_v1.pt` | Model weight file |
| `DEFAULT_CONFIDENCE` | 0.25 | Default detection threshold |
| `DEFAULT_MASK_PADDING` | 10 | Default padding in pixels |

---

## 10. Performance Benchmarks

Tested on 16 real product images (mixed categories: electronics, textiles, food, machinery).

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

## 11. Limitations & Future Roadmap

### 11.1 Current Limitations

| Limitation | Description |
|------------|-------------|
| **Bounding-box masks only** | Masks are rectangular, which may include non-watermark pixels in the inpainting target. Pixel-level segmentation (e.g., SAM) would be more precise. |
| **Single detection model** | Relies on one YOLO model; unusual watermark styles may be missed. |
| **No GPU parallelism** | Processes images sequentially; could benefit from batch GPU inference. |
| **Large watermark inpainting** | When watermarks cover >30% of the image, LaMa may produce blurry or inconsistent fills. |
| **No UI** | Currently CLI-only. A Gradio web UI is planned. |

### 11.2 Future Improvements Under Consideration

| Improvement | Description | Status |
|-------------|-------------|--------|
| **Gradio Web UI** | Interactive UI for single/batch processing with model selection | Planned |
| **SAM-based mask refinement** | Use Segment Anything Model to generate pixel-precise masks from bbox prompts | Research |
| **YOLO11x upgrade** | Newer YOLO11 architecture (e.g., `corzent/yolo11x_watermark_detection`) for better detection | Research |
| **Multi-pass inpainting** | Iteratively inpaint large watermarks in smaller chunks for better quality | Research |
| **Self-supervised models** | Models like SWCNN, SSNet, PSLNet that don't need explicit masks | Research |
| **Academic models** | SLBR (ACM MM 2021), deep-blind-watermark-removal (AAAI 2021) for end-to-end blind removal | Research |

---

*Document generated for the Pepagora Watermark Remover project.*
