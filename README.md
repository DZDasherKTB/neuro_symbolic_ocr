# Neuro-Symbolic Pipeline for Handwritten Text Recognition (HTR)

## Project Overview:

This project develops a state-of-the-art OCR pipeline where LLMs and VLMs are not merely post-processors but active controllers of the ingestion, recognition, and validation cycles. By integrating **Qwen2.5-VL** and  **TrOCR** , we address the primary failure points of handwriting OCR: inconsistent segmentation and lack of semantic context.

## Detailed Technical Pipeline:

### Stage 1: VLM-Driven Document Ingestion & Normalization

Instead of "blind" grayscale conversion, we use a **Vision Encoder (ViT-L)** to generate a document profile.

* **Adaptive Binarization:** The VLM identifies "low-contrast" vs. "high-noise" regions. For low-contrast areas (faded ink), we apply  **Sauvola’s local thresholding** ; for high-noise (scanner artifacts), we apply  **Median Filtering** .
* **Stroke Normalization:** Using an estimated stroke width **$w$**, we rescale the image **$I$** by a factor **$s = 10/w$** to normalize all text to a 10px average stroke thickness, reducing the variance for the subsequent Transformer encoder.

### Stage 2: Multi-Scale Adaptive Tiling (MSAT)

To prevent the "Cutting Word" problem where a bounding box splits a cursive ligature:

* **Mechanism:** We generate a dual-stream input.
  * **Global Stream:** Low-res full page for layout analysis.
  * **Local Stream:** High-res overlapping tiles (**$224 \times 224$** or **$384 \times 384$**).
* **Contextual Overlap:** Tiles are generated with a 20% stride overlap. A lightweight **LLM agent** decides if a tile boundary intersects a high-confidence text region; if so, it shifts the tile window to preserve word integrity.

### Stage 3: Hierarchical Text Detection & Bounding Boxes

* **Detector:** **DBNet++** with a ResNet-50 backbone. It produces a probability map of text kernels.
* **VLM Refinement:** For boxes with a confidence score **$C < 0.6$**, the image crop is sent to a **Qwen2-VL-2B** model with the prompt: *"Are there characters in this crop?"* This significantly improves recall on faint marginalia.
* **Output:** Word-level coordinates **$[x_1, y_1, x_2, y_2]$** and a "Style Tag" (e.g.,  *cursive, printed, numeric* ).

### Stage 4: Transformer-Based Recognition & Lattice Generation

* **Core Model:**  **TrOCR-Large-Handwritten** .
* **Encoding:** The ViT-Large encoder extracts visual features from word crops.
* **Probabilistic Decoding:** We modify the decoding head to output a  **Token Lattice** . Instead of a greedy search (Top-1), we extract the Top-K candidates for every word.
  * **Example Output:**
    * **$W_1$**: `{"The": 0.98, "Tie": 0.01}`
    * **$W_2$**: `{"quick": 0.70, "qu1ck": 0.25, "quack": 0.05}`
    * **$W_3$**: `{"brown": 0.85, "brawn": 0.12}`

### Stage 5: LLM Sentence Reconstruction & Probability Optimization

The LLM (e.g.,  **Llama-3.1-8B** ) acts as a "Neural Beam Search" over the lattice.

* **Input:** "Context: {Previous Sentences} | Options: [W1, W2, W3...]"
* **Objective Function:** The LLM selects a sequence **$Y$** that maximizes:

  $$
  P(Y) = \lambda \cdot P_{Visual}(Y|I) + (1-\lambda) \cdot P_{Language}(Y)
  $$

  where **$\lambda$** is a weight tuned during the validation phase to balance visual evidence vs. grammatical likelihood.

### Stage 6: VLM Visual Validation (The Hallucination Guard)

To prevent the LLM from "over-correcting" (e.g., changing a person's name because it isn't in the dictionary):

* **Logic:** If the LLM correction differs from the Top-1 OCR result, the **VLM Validator** (Qwen2.5) is triggered.
* **Task:** It performs a  **Zero-shot Cross-Modal Match** .
  * *Prompt:* "Crop [Image] contains text 'brawn'. Is this visually plausible?"
  * *Action:* If 'No', the system reverts to the OCR candidate.

---

## Evaluation & Datasets

### Metrics for Success

1. **Character Error Rate (CER):** **$CER = \frac{Edit\_Distance(GT, Pred)}{Total\_Chars}$**.
2. **Word Error Rate (WER):** Target reduction of 15% over baseline TrOCR.
3. **Inference Latency:** Measuring seconds/page to ensure the pipeline is practical.

### Benchmarking Baselines

| **System**       | **Strengths**             | **Weaknesses**                                 |
| ---------------------- | ------------------------------- | ---------------------------------------------------- |
| **Tesseract v5** | Fast, Lightweight               | Poor on cursive/handwriting                          |
| **PaddleOCR**    | Great Detection                 | Weak semantic correction                             |
| **TrOCR (Base)** | High Accuracy                   | No global context; Hallucinates                      |
| **Our Proposal** | **Hybrid Vision/Context** | **Higher Compute (Mitigated by Quantization)** |

### Training & Finetuning Strategy

1. **Stage A (Vision):** Finetune **TrOCR** on **IAM Handwriting Database** and **CVL** for 10 epochs using AdamW optimizer.
2. **Stage B (Correction):** Finetune **Llama-3.1** using **LoRA** (Rank=8) on a "Corrupted Text" dataset where we simulate OCR errors in common sentences.
3. **Stage C (Validation):** Train a linear head on the VLM's projection layer to classify `Correct/Incorrect` pairs of (Image, Text).

## Research Contributions & Finetuning

1. **OCR Finetuning:** LoRA-based adaptation of **TrOCR** on **IAM** and **CVL** datasets, specifically targeting historical ligatures.
2. **LLM "Correction" Finetuning:** Training the LLM on synthetic OCR error pairs (Noisy Text **$\rightarrow$** Ground Truth) to learn common handwriting misread patterns.
3. **VLM "Validation" Finetuning:** A discriminative task training the VLM to output a binary `Match/Mismatch` score for (Image Crop, Text String) pairs.
