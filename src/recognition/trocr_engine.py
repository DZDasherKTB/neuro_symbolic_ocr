import torch
import cv2
from PIL import Image
from typing import List, Dict, Any
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from src.pipeline.config_loader import ConfigLoader


class TrOCREngine:

    def __init__(self):

        config = ConfigLoader()
        self.cfg = config.models.recognition
        self.device = config.pipeline.device

        print(f"Loading TrOCR model: {self.cfg.model_path}")

        self.processor = TrOCRProcessor.from_pretrained(self.cfg.model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(
            self.cfg.model_path
        ).to(self.device)

        self.model.eval()

        self.max_length = self.cfg.max_length
        self.beam_size = getattr(self.cfg, "beam_size", 4)

    # -------------------------------------------------------

    def recognize_lines(self, line_crops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Runs OCR on line-level crops.

        Input:
            line_crops = [
                {
                    paragraph_id
                    line_id
                    image
                    bbox
                }
            ]

        Output:
            recognized_lines = [
                {
                    paragraph_id
                    line_id
                    text
                    bbox
                }
            ]
        """

        recognized_lines = []

        for item in line_crops:

            line_img = item["image"]

            pil_img = self._to_pil(line_img)

            pixel_values = self.processor(
                pil_img,
                return_tensors="pt"
            ).pixel_values.to(self.device)

            with torch.no_grad():

                generated_ids = self.model.generate(
                    pixel_values,
                    num_beams=self.beam_size,
                    max_length=self.max_length
                )

            text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

            recognized_lines.append({
                "block_id": item.get("block_id", 0),
                "line_id": item["line_id"],
                "bbox": item["bbox"],
                "text": text.strip()
            })

        return recognized_lines

    def _to_pil(self, img):

        if len(img.shape) == 2:
            return Image.fromarray(img).convert("RGB")

        return Image.fromarray(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )