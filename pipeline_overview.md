# Neuro-Symbolic Handwritten Text Recognition Pipeline

## Overview

This project explores a  **neuro-symbolic OCR architecture for handwritten documents** , combining deep learning models for perception with structured reasoning and contextual reconstruction.

Traditional OCR pipelines rely purely on neural recognition. However, handwritten text recognition often produces noisy outputs. This system introduces **symbolic structuring and language-based reasoning** to stabilize predictions.

The pipeline integrates:

* Vision models for detection and recognition
* Structured token lattices for symbolic representation
* Language models for contextual reconstruction
* Multimodal validation to reduce hallucination

The system is designed to operate under  **limited hardware resources (~6GB VRAM)** , requiring careful model selection, tiling strategies, and memory-efficient inference.

---

# Full System Pipeline

<pre class="overflow-visible! px-0!" data-start="1299" data-end="1797"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>Document Image</span><br/><span>      │</span><br/><span>      ▼</span><br/><span>1. Document Profiling (VLM)</span><br/><span>      │</span><br/><span>      ▼</span><br/><span>2. Adaptive Preprocessing</span><br/><span>      │</span><br/><span>      ▼</span><br/><span>3. Multi-Scale Adaptive Tiling</span><br/><span>      │</span><br/><span>      ▼</span><br/><span>4. Text Detection</span><br/><span>      │</span><br/><span>      ▼</span><br/><span>5. Bounding Box Refinement</span><br/><span>      │</span><br/><span>      ▼</span><br/><span>6. Word / Line Cropping</span><br/><span>      │</span><br/><span>      ▼</span><br/><span>7. OCR Recognition</span><br/><span>      │</span><br/><span>      ▼</span><br/><span>8. Structured Lattice Construction</span><br/><span>      │</span><br/><span>      ▼</span><br/><span>9. Contextual Reconstruction (LLM)</span><br/><span>      │</span><br/><span>      ▼</span><br/><span>10. Multimodal Validation (VLM)</span><br/><span>      │</span><br/><span>      ▼</span><br/><span>Final Reconstructed Text</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

# 1. Document Profiling

Goal: Estimate document characteristics before processing.

This stage uses a **Vision-Language Model (VLM)** to estimate:

* script style (print vs cursive)
* degradation level
* background noise
* stroke thickness
* contrast distribution

These signals guide preprocessing and tiling decisions.

Example outputs:

<pre class="overflow-visible! px-0!" data-start="2145" data-end="2257"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>script_type = cursive_handwriting</span><br/><span>noise_level = low</span><br/><span>document_condition = clean</span><br/><span>stroke_variation = medium</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

The profiling stage acts as a **control module** for later pipeline stages.

---

# 2. Adaptive Preprocessing

Preprocessing attempts to normalize handwriting variability before recognition.

Techniques explored:

### Stroke Width Normalization

Used to stabilize pen stroke thickness.

Approaches tested:

* morphological closing
* adaptive dilation
* stroke width transform approximation

Goal:

<pre class="overflow-visible! px-0!" data-start="2656" data-end="2725"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>thin strokes → normalized width</span><br/><span>thick strokes → reduced noise</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

### Illumination and Contrast Normalization

Methods evaluated:

* CLAHE (Contrast Limited Adaptive Histogram Equalization)
* gamma correction
* bilateral filtering

These techniques stabilize the foreground text against background noise.

---

### Noise Suppression

Applied when document profiler indicates noise.

Filters tested:

* median filter
* bilateral filter
* non-local means denoising

---

# 3. Adaptive Multi-Scale Tiling

Handwritten documents can be large relative to available GPU memory.

To handle this, images are processed using  **adaptive tiling** .

Configuration example:

<pre class="overflow-visible! px-0!" data-start="3328" data-end="3389"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>512 × 512 tiles</span><br/><span>768 × 768 tiles</span><br/><span>stride overlap 20–30%</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

Advantages:

* prevents GPU memory overflow
* preserves local text structure
* improves detection coverage

Tiles are later merged using  **global coordinate mapping** .

---

# 4. Text Detection

Goal: locate text regions in each tile.

Current approach:

YOLOv8 handwritten text detector

Experiments include:

* confidence threshold tuning
* NMS threshold adjustments
* stride and receptive field tuning

Outputs:

<pre class="overflow-visible! px-0!" data-start="3807" data-end="3861"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>bounding box</span><br/><span>confidence score</span><br/><span>tile coordinates</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

Detections are converted back into  **global image coordinates** .

---

# 5. Bounding Box Refinement

Detected boxes are refined using heuristics:

### Box merging

Overlapping detections merged using IoU.

### Geometric normalization

Boxes expanded slightly to capture full characters.

### NMS filtering

Duplicate detections removed.

---

# 6. Text Region Cropping

Two strategies explored:

### Word-Level Cropping

Each bounding box becomes a word candidate.

Pros:

* fine granularity

Cons:

* OCR errors accumulate
* context lost

---

### Line-Level Cropping (Planned Transition)

Bounding boxes grouped into lines based on vertical overlap.

Line crops are then extracted:

<pre class="overflow-visible! px-0!" data-start="4543" data-end="4630"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>x1 = min(all boxes)</span><br/><span>y1 = min(all boxes)</span><br/><span>x2 = max(all boxes)</span><br/><span>y2 = max(all boxes)</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

This significantly improves OCR stability.

---

# 7. OCR Recognition

Recognition currently uses:

**TrOCR (Transformer OCR)**

Architecture:

<pre class="overflow-visible! px-0!" data-start="4776" data-end="4830"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>Vision Transformer encoder</span><br/><span>Transformer decoder</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

Inference strategy:

* beam search
* candidate token extraction
* confidence scoring

Outputs per word:

<pre class="overflow-visible! px-0!" data-start="4937" data-end="4992"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>candidate tokens</span><br/><span>visual confidence</span><br/><span>bounding box</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

# 8. Structured Lattice Construction

OCR outputs are organized into a  **token lattice** .

Example structure:

<pre class="overflow-visible! px-0!" data-start="5110" data-end="5189"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>Line 1:</span><br/><span>[An] [attempt] [to] [get]</span><br/><br/><span>Line 2:</span><br/><span>[more] [information] [about]</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

In earlier versions, multiple candidate tokens were stored:

<pre class="overflow-visible! px-0!" data-start="5252" data-end="5317"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>word:</span><br/><span>  candidates = ["an", "a", "n"]</span><br/><span>  confidence = 0.92</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

However large candidate sets produced unstable LLM prompts.

---

# 9. Contextual Reconstruction

A lightweight language model is used to reconstruct coherent text.

Model variants tested:

* Phi-3 Mini
* Qwen-2.5 1.5B
* smaller instruction-tuned models

Key design goal:

Use **minimal token context reconstruction** rather than free generation.

Approach:

<pre class="overflow-visible! px-0!" data-start="5678" data-end="5745"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>OCR lines</span><br/><span>→ grammar correction</span><br/><span>→ contextual spelling repair</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

This stage is being redesigned to operate on  **line-level text rather than token lattices** .

---

# 10. Multimodal Validation

A VLM is used as a  **hallucination guard** .

The model compares:

<pre class="overflow-visible! px-0!" data-start="5941" data-end="5983"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>reconstructed text</span><br/><span>vs</span><br/><span>image region</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

If semantic mismatch occurs:

* confidence reduced
* reconstruction flagged

This stage helps reduce LLM hallucination risk.

---

# Training and Benchmarking

Evaluation datasets:

IAM Handwriting Dataset

GOTHAM Handwriting Dataset

Metrics used:

<pre class="overflow-visible! px-0!" data-start="6236" data-end="6329"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>CER — Character Error Rate</span><br/><span>WER — Word Error Rate</span><br/><span>Detection Recall</span><br/><span>Detection Precision</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

# Hardware Constraints

Development performed on:

<pre class="overflow-visible! px-0!" data-start="6387" data-end="6409"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>GPU VRAM: 6 GB</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

This required:

* quantized language models
* tiled image inference
* memory-efficient architectures

Model experiments prioritize  **low-VRAM compatibility** .

---

# Current Limitations

Observed challenges:

* word-level recognition unstable for cursive handwriting
* detection misses some text regions
* token lattices produce overly complex LLM prompts
* OCR confidence noise propagates to reconstruction stage

---

# Planned Improvements

Pipeline improvements in progress:

### Line-Level OCR

Switch recognition to full line crops.

### Improved Detection

Tune YOLO detection thresholds and anchor sizes.

### Lightweight Language Models

Evaluate models with smaller context and token requirements.

### Context-Aware Reconstruction

Limit LLM role to grammar correction rather than full reconstruction.

### Optional Model Fine-Tuning

Potential fine-tuning on IAM dataset.
