# Research Report: Next-Level Improvements for Watermark Remover

## Current System Baseline
- **Detection:** YOLOv8-small (mnemic/watermarks_yolov8, 22.5 MB)
- **Inpainting:** LaMa big-lama (196 MB)
- **Mask:** Rectangular bounding-box with padding

---

## AREA 1: Better Watermark Detection

### Upgrade: YOLO11x Watermark Detector

| Attribute | Current (YOLOv8-s) | Upgrade (YOLO11x) |
|---|---|---|
| Model | mnemic/watermarks_yolov8 | corzent/yolo11x_watermark_detection |
| Architecture | YOLOv8-small | YOLO11x (latest generation) |
| Input size | default 640 | 1280×1280 (2x more detail) |
| mAP@50 | Unknown | 0.900 |
| Precision | Unknown | 0.896 |
| Recall | Unknown | 0.883 |
| Downloads | N/A | 2,388/month |
| Integration | YOLO("path") | YOLO("corzent/yolo11x_watermark_detection") — drop-in replacement |

**Verdict: HIGH PRIORITY** — This is a drop-in upgrade. Same Ultralytics API, newer architecture, higher-resolution input (1280 vs 640), published metrics proving strong performance. The 1280px input is critical for catching small watermarks that our current 640px model may miss.

**Effort:** Minimal. Change one line of code.

---

## AREA 2: Better Mask Generation (Biggest Impact)

### Upgrade: SAM 2 (Segment Anything Model) for Pixel-Precise Masks

This is the single biggest improvement available. Our current system uses rectangular bounding boxes as masks, meaning we inpaint non-watermark pixels too. SAM can generate pixel-precise segmentation masks from bbox prompts.

| Attribute | Current (bbox) | Upgrade (SAM) |
|---|---|---|
| Mask shape | Rectangle | Pixel-precise contour |
| Unnecessary inpainting | Corners of bbox (wasted area) | Only watermark pixels |
| Architecture | cv2.rectangle() | ViT-based foundation model (1.1B masks trained) |
| Checkpoint sizes | None | ViT-H: 2.4 GB, ViT-L: 1.2 GB, ViT-B: 375 MB |
| License | N/A | Apache 2.0 |

**How it works in our pipeline:**

**Key API:**

**Verdict: HIGHEST PRIORITY** — This transforms rectangular masks into precise watermark-shape masks. Less area inpainted = better quality output. SAM is the gold standard for this.

**Effort:** Medium. Add 20 lines, download SAM ViT-B checkpoint (375 MB).

---

## AREA 3: Better Inpainting Model

### Option A: MAT (Mask-Aware Transformer) — CVPR 2022 Best Paper Finalist

| Attribute | LaMa (current) | MAT |
|---|---|---|
| Architecture | Fourier Convolution ResNet | Transformer-based |
| Paper | WACV 2022 | CVPR 2022 (Best Paper Finalist) |
| Strength | Large masks, resolution-robust | Large holes, high fidelity + diversity |
| Stars | ~7k (LaMa repo) | 967 |
| Pre-trained | Places (general scenes) | CelebA-HQ, FFHQ, Places-512 |
| Input constraint | Any resolution | Must be multiple of 512 |
| Dependencies | simple-lama-inpainting (clean) | StyleGAN2-ADA based (heavy, Python 3.7, CUDA 11.0) |
| Benchmark vs LaMa | FID: 0.99 (1.8M data) | FID: 0.78 (8M data) — 21% better |

MAT outperforms LaMa on Places benchmark (FID 0.78 vs 0.99 on 8M data). However:

- Requires Python 3.7, PyTorch 1.7.1, CUDA 11.0 — significant backward compatibility issue
- Resolution must be multiples of 512 — requires resize/pad logic
- No simple pip package — must clone and integrate manually
- Heavy CUDA dependency (custom C++/CUDA ops from StyleGAN2)

**Verdict: LOW PRIORITY for now** — Better output quality, but the integration cost and dependency constraints are high. Worth revisiting when we containerize the system. Our LaMa already performs well, and SAM mask improvements will have more impact.

### Option B: Keep LaMa (Recommended)

LaMa remains an excellent choice because:

- Resolution-robust (no resize needed)
- Clean pip-installable package
- Handles diverse image types well
- SAM mask improvement will make LaMa perform even better (less area to fill)

---

## AREA 4: End-to-End Academic Models (Blind Watermark Removal)

These models attempt detection + removal in one shot, without needing a separate detector.

### SLBR — Self-calibrated Localization and Background Refinement (ACM MM 2021)
- 250 stars, 42 forks
- Multi-task: detects watermark + refines background simultaneously
- Pre-trained model available (Google Drive / OneDrive, trained on CLWD dataset)
- **Limitation:** Trained at 256×256 — will lose detail on our product images
- **Limitation:** 5 years old, no updates
- PyTorch ≥1.0, standard dependencies

### deep-blind-watermark-removal — SplitNet (AAAI 2021)
- 259 stars, 59 forks
- Two-stage: SplitNet (detect + remove) → RefineNet (smooth + refine)
- Pre-trained model available (Google Drive, 27kpng_model_best.pth.tar)
- Has Google Colab demo
- Datasets on HuggingFace (vinthony/watermark-removal-logo)
- **Limitation:** 6 years old, single contributor
- **Limitation:** Trained on synthetic watermarks — may not generalize to diverse real-world styles

### SWCNN — Self-supervised CNN (IEEE TCSVT 2024)
- 74 stars — newest of the three
- Self-supervised: doesn't need paired data (watermarked + clean)
- Uses heterogeneous U-Net + mixed loss for texture preservation
- Pre-trained model available (Google Drive + Baidu)
- **Limitation:** Designed for semi-transparent watermarks with known alpha
- **Limitation:** Requires `--alpha 0.3` parameter — assumes watermark transparency is known

### PSLNet — Perceptive Self-supervised Learning (IEEE TCSVT 2024)
- 22 stars — very recent
- Handles both noise AND watermarks simultaneously
- Parallel dual-network architecture
- **Limitation:** Extremely niche (noisy + watermarked images)
- **Limitation:** Requires noise level + transparency parameters

**Verdict for all end-to-end models: NOT RECOMMENDED as primary system**

- All trained at low resolution (256×256) — unusable for product images without quality loss
- Our two-stage approach (detect independently + inpaint independently) gives more control
- However, SLBR or SplitNet could be useful as a **secondary verification model** — run after inpainting to check if any watermark remnants remain

---

## AREA 5: Multi-Pass & Post-Processing Enhancement

### Strategy A: Iterative Detection + Inpainting
Run the pipeline twice: after first inpainting, re-run detection to catch any residual watermark traces, then inpaint again.

### Strategy B: Image Quality Enhancement
After inpainting, apply a lightweight super-resolution or sharpening model to fix any slight blur in the inpainted region:

- **Real-ESRGAN** for targeted upscaling of the inpainted bbox region only
- **CodeFormer** if product images contain faces

### Strategy C: Confidence-Based Routing
Use detection confidence to route processing:

- **High confidence (>0.7):** Standard pipeline
- **Medium confidence (0.3-0.7):** Use SAM for precise masking + larger padding
- **Low confidence (<0.3):** Flag for human review

---

## RECOMMENDED UPGRADE PATH (Priority Order)

| Priority | Upgrade | Impact | Effort | Status |
|---|---|---|---|---|
| 1 | SAM mask refinement | Very High — pixel-precise masks | Medium (~20 lines + 375MB model) | Ready to implement |
| 2 | YOLO11x detector | High — better detection accuracy | Minimal (1 line change) | Ready to implement |
| 3 | Multi-pass pipeline | Medium — catches residual marks | Low (~15 lines) | Ready to implement |
| 4 | Confidence routing | Medium — smarter threshold logic | Low (~10 lines) | Ready to implement |
| 5 | MAT inpainting | Medium-High quality improvement | High (dependency hell) | Future (needs containerization) |
| 6 | SLBR as verifier | Low-Medium — secondary check | Medium (custom integration) | Future |

### Priority 1+2 Combined Architecture (Next-Level System):

This upgrades all three weak points of the current system:

- **Better detection** (YOLO11x, 1280px input, mAP 0.9)
- **Better masks** (SAM pixel-precise contours instead of rectangles)
- **Better coverage** (second-pass catches anything missed)

> Shall I implement any of these upgrades?
>
> Completed: Compile final recommendations (5/5)
