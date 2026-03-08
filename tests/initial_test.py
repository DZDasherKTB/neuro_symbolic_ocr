import cv2
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

from src.reasoning.llm_reconstruction import LLMReconstructor

from src.validation.vlm_validator import VLMValidator


def run_pipeline(image_path: str):

    print("Loading configs...")
    config = ConfigLoader()

    print("Initializing modules...")

    profiler = DocumentProfiler()
    preprocessor = AdaptivePreprocessor()
    normalizer = StrokeNormalizer()

    tiler = AdaptiveTiler()

    detector = DBNetDetector()
    refiner = BBoxRefiner()

    trocr = TrOCREngine()
    lattice_builder = LatticeBuilder()

    llm = LLMReconstructor()
    vlm = VLMValidator()

    print("\n--- Stage 1: Document Profiling ---")
    profile = profiler.profile(image_path)
    print("Profile:", profile.__dict__)

    print("\n--- Stage 2: Preprocessing ---")
    processed = preprocessor.process(image_path, profile)
    normalized = normalizer.normalize(processed)

    print("\n--- Stage 3: Tiling ---")
    tiles = tiler.generate(normalized)
    print(f"Tiles generated: {len(tiles)}")

    print("\n--- Stage 4: Detection ---")
    detections = detector.detect(tiles)
    detections = refiner.refine(normalized, detections)

    print(f"Detections: {len(detections)}")

    print("\n--- Stage 5: Word Cropping ---")
    word_crops = BBoxUtils.crop_words(normalized, detections)
    print(f"Word crops: {len(word_crops)}")

    print("\n--- Stage 6: OCR Recognition ---")
    lattice_data = trocr.recognize_to_lattice(word_crops)

    print("\n--- Stage 7: Lattice Structuring ---")
    structured_lattice = lattice_builder.build_structured_lattice(lattice_data)

    print("\n--- Stage 8: LLM Reconstruction ---")
    reconstructed_text = llm.reconstruct(structured_lattice)

    print("LLM Output:")
    print(reconstructed_text)

    print("\n--- Stage 9: VLM Validation ---")
    final_text = vlm.validate(reconstructed_text, word_crops, lattice_data)

    print("\n===== FINAL TEXT =====")
    print(final_text)

    return final_text


if __name__ == "__main__":

    image_path = input("Enter path to test image: ")

    text = run_pipeline(image_path)

    print("\nPipeline completed.")