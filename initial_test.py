import cv2
import torch

from src.pipeline.config_loader import ConfigLoader

from src.ingestion.document_profiler import DocumentProfiler
from src.ingestion.adaptive_preprocessor import AdaptivePreprocessor
from src.ingestion.stroke_normalizer import StrokeNormalizer

from src.tiling.adaptive_tiler import AdaptiveTiler

from src.detection.dbnet_detector import DBNetDetector
from src.detection.bbox_refiner import BBoxRefiner
from src.detection.bbox_utils import BBoxUtils

from src.recognition.trocr_engine import TrOCREngine
from src.recognition.lattice_builder import LatticeBuilder
from src.recognition.line_cropper import LineCropper

from src.reasoning.llm_reconstruction import LLMReconstructor
from src.validation.vlm_validator import VLMValidator


def run_pipeline(image_path: str):

    print("Loading configs...")
    config = ConfigLoader()

    print("\n--- Stage 1: Document Profiling ---")
    profiler = DocumentProfiler()
    profile = profiler.profile(image_path)
    print("Profile:", profile.__dict__)

    # free profiler
    del profiler
    torch.cuda.empty_cache()

    print("\n--- Stage 2: Preprocessing ---")
    preprocessor = AdaptivePreprocessor()
    normalizer = StrokeNormalizer()

    processed = preprocessor.process(image_path, profile)
    normalized = normalizer.normalize(processed)

    del preprocessor
    del normalizer
    torch.cuda.empty_cache()

    print("\n--- Stage 3: Tiling ---")
    tiler = AdaptiveTiler()
    tiles = tiler.generate(normalized)

    print(f"Tiles generated: {len(tiles)}")

    del tiler
    torch.cuda.empty_cache()

    print("\n--- Stage 4: Detection ---")
    detector = DBNetDetector()
    refiner = BBoxRefiner()

    detections = detector.detect(tiles)
    detections = refiner.refine(normalized, detections)

    print(f"Detections: {len(detections)}")

    del detector
    del refiner
    torch.cuda.empty_cache()

    # print("\n--- Stage 5: Line Cropping ---")

    # line_crops = LineCropper.crop_lines(normalized, detections)

    # print(f"Line crops: {len(line_crops)}")


    # print("\n--- Stage 6: Line OCR ---")

    # line_texts = []

    # for line in line_crops:

    #     text = trocr.recognize_line(line["image"])

    #     line_texts.append(text)

    # print(line_texts)
    print("\n--- Stage 5: Word Cropping ---")
    word_crops = BBoxUtils.crop_words(normalized, detections)
    print(f"Word crops: {len(word_crops)}")

    print("\n--- Stage 6: OCR Recognition ---")
    trocr = TrOCREngine()

    lattice_data = trocr.recognize_to_lattice(word_crops)

    del trocr
    torch.cuda.empty_cache()

    print("\n--- Stage 7: Lattice Structuring ---")
    lattice_builder = LatticeBuilder()

    structured_lattice = lattice_builder.build_structured_lattice(lattice_data)

    del lattice_builder
    torch.cuda.empty_cache()

    print("\n--- Stage 8: LLM Reconstruction ---")
    llm = LLMReconstructor()

    reconstructed_text = llm.reconstruct(structured_lattice)

    print("LLM Output:")
    print(reconstructed_text)

    del llm
    torch.cuda.empty_cache()

    print("\n--- Stage 9: VLM Validation ---")
    vlm = VLMValidator()

    final_text = vlm.validate(reconstructed_text, word_crops, lattice_data)

    print("\n===== FINAL TEXT =====")
    print(final_text)

    del vlm
    torch.cuda.empty_cache()

    return final_text


if __name__ == "__main__":

    text = run_pipeline("data/raw/image.png")

    print("\nPipeline completed.")