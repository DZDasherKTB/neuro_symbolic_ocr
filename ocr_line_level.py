import torch
import cv2

from src.pipeline.config_loader import ConfigLoader

from src.ingestion.document_profiler import DocumentProfiler
from src.ingestion.adaptive_preprocessor import AdaptivePreprocessor
from src.ingestion.stroke_normalizer import StrokeNormalizer

from src.tiling.adaptive_tiler import AdaptiveTiler

from src.detection.dbnet_detector import DBNetDetector

from src.recognition.line_cropper import LineCropper
from src.layout.line_segmenter import LineSegmenter


from src.recognition.trocr_engine import TrOCREngine

from src.reasoning.llm_reconstruction import LLMReconstructor

from src.validation.vlm_validator import VLMValidator


def run_pipeline(image_path: str):

    print("Loading configs...")
    config = ConfigLoader()

    # --------------------------------------------------
    print("\n--- Stage 1: Document Profiling ---")

    profiler = DocumentProfiler()
    profile = profiler.profile(image_path)

    print("Profile:", profile.__dict__)

    del profiler
    torch.cuda.empty_cache()

    # --------------------------------------------------
    print("\n--- Stage 2: Preprocessing ---")

    preprocessor = AdaptivePreprocessor()
    normalizer = StrokeNormalizer()

    processed = preprocessor.process(image_path, profile)
    normalized = normalizer.normalize(processed)

    del preprocessor
    del normalizer
    torch.cuda.empty_cache()

    # --------------------------------------------------
    print("\n--- Stage 3: Adaptive Tiling ---")

    tiler = AdaptiveTiler()
    tiles = tiler.generate(normalized)

    print(f"Tiles generated: {len(tiles)}")

    del tiler
    torch.cuda.empty_cache()

    print("\n--- Stage 4: Detection ---")

    detector = DBNetDetector()

    detections = detector.detect(tiles)

    text_bbox = detector.detect_text_region(detections)

    if text_bbox is None:
        raise RuntimeError("No text detected")

    x1, y1, x2, y2 = text_bbox

    text_region = normalized[y1:y2, x1:x2]

    print("Text region:", text_bbox)

    del detector
    torch.cuda.empty_cache()

    # --------------------------------------------------
    print("\n--- Stage 5: Line Segmentation (Projection) ---")

    segmenter = LineSegmenter()

    line_crops = segmenter.extract_line_crops(text_region)

    print("Lines detected:", len(line_crops))

    # --------------------------------------------------
    print("\n--- Stage 6: OCR Recognition ---")

    trocr = TrOCREngine()

    recognized_lines = trocr.recognize_lines(line_crops)

    del trocr
    torch.cuda.empty_cache()

    print("\nOCR Output:")
    for l in recognized_lines:
        print(l["text"])

    # --------------------------------------------------
    print("\n--- Stage 7: LLM Reconstruction ---")

    llm = LLMReconstructor()

    reconstructed_text = llm.reconstruct(recognized_lines)

    print("\nLLM Corrected Text:")
    print(reconstructed_text)

    del llm
    torch.cuda.empty_cache()

    # --------------------------------------------------
    print("\n--- Stage 8: VLM Validation ---")

    vlm = VLMValidator()

    final_text = vlm.validate(
        reconstructed_text,
        line_crops,
        recognized_lines
    )

    del vlm
    torch.cuda.empty_cache()

    # --------------------------------------------------
    print("\n===== FINAL TEXT =====\n")
    print(final_text)

    return final_text


if __name__ == "__main__":

    image_path = "data/raw/image.png"

    text = run_pipeline(image_path)

    print("\nPipeline completed.")