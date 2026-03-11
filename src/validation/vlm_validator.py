import torch
from typing import List, Dict
from PIL import Image
import cv2
from src.pipeline.config_loader import ConfigLoader


class VLMValidator:

    def __init__(self, vlm_model=None):

        config = ConfigLoader()
        self.cfg = config.models.validation
        self.device = config.pipeline.device

        self.vlm = vlm_model

    # ---------------------------------------------------------

    def validate(
        self,
        corrected_text: str,
        line_crops: List[Dict],
        recognized_lines: List[Dict]
    ) -> str:
        """
        Checks if LLM output is consistent with visual evidence.
        If hallucinations are detected, revert to raw OCR line.
        """

        if self.vlm is None:
            return corrected_text

        corrected_lines = corrected_text.split("\n")

        validated_lines = []

        for i, line in enumerate(corrected_lines):

            if i >= len(line_crops):
                break

            crop = line_crops[i]["image"]

            if self._verify_line(crop, line):

                validated_lines.append(line)

            else:
                validated_lines.append(recognized_lines[i]["text"])

        return "\n".join(validated_lines)

    def _verify_line(self, image, text):

        pil_img = self._to_pil(image)

        prompt = f"""
Does the following text match the handwritten content in the image?

Text:
{text}

Answer only YES or NO.
"""

        response = self.vlm.generate_response(pil_img, prompt)

        if response is None:
            return True

        return "yes" in response.lower()

    def _to_pil(self, img):

        if len(img.shape) == 2:
            return Image.fromarray(img).convert("RGB")

        return Image.fromarray(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )