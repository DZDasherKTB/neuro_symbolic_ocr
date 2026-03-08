import torch
from PIL import Image
from typing import List, Dict, Any
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from src.pipeline.config_loader import ConfigLoader
import cv2
class TrOCREngine:
    def __init__(self):
        config = ConfigLoader()
        self.cfg = config.models.recognition
        self.device = config.pipeline.device
        
        print(f"Initializing TrOCR Recognition: {self.cfg.model_path}")
        self.processor = TrOCRProcessor.from_pretrained(self.cfg.model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.cfg.model_path).to(self.device)
        self.model.eval()        
        self.beam_size = self.cfg.beam_size
        self.top_k = self.cfg.lattice.top_k

    def recognize_to_lattice(self, word_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes word crops and generates top-k token candidates with log-probabilities.
        """
        lattice_data = []

        for item in word_data:
            crop = item['image']
            if len(crop.shape) == 2:
                pil_img = Image.fromarray(crop).convert("RGB")
            else:
                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values.to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    num_beams=self.beam_size,
                    num_return_sequences=self.top_k,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_length=self.cfg.max_length
                )

            candidates = []
            scores = torch.exp(outputs.sequences_scores) if hasattr(outputs, 'sequences_scores') else [1.0] * self.top_k

            for i, sequence in enumerate(outputs.sequences):
                text = self.processor.batch_decode(sequence.unsqueeze(0), skip_special_tokens=True)[0]
                candidates.append({
                    "text": text,
                    "prob": float(scores[i])
                })

            lattice_data.append({
                "candidates": candidates,
                "bbox": item['bbox'],
                "confidence_ocr": item['confidence'],
                "parent_tile_id": item['parent_tile_id']
            })

        return lattice_data