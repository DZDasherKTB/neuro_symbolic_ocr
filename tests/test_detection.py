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

image_path = "../data/raw/BenthamDatasetR0-Images.tbz"

print("Loading configs...")
config = ConfigLoader()

print("Initializing modules...")

profiler = DocumentProfiler()
preprocessor = AdaptivePreprocessor()
normalizer = StrokeNormalizer()

tiler = AdaptiveTiler()

detector = DBNetDetector()
refiner = BBoxRefiner()

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