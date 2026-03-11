import torch
from typing import List, Dict
from PIL import Image
import cv2
from transformers import AutoProcessor, AutoModelForVision2Seq
from src.pipeline.config_loader import ConfigLoader


class VLMValidator:

    def __init__(self):

        config = ConfigLoader()

        self.cfg = config.models.validation
        self.device = config.pipeline.device

        model_name = config.models.vlm.model_name

        print(f"Loading VLM for validation: {model_name}")

        self.processor = AutoProcessor.from_pretrained(model_name)

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)

        self.model.eval()

    def validate(
        self,
        corrected_text: str,
        line_crops: List[Dict],
        recognized_lines: List[Dict]
    ) -> str:

        corrected_lines = corrected_text.split("\n")

        validated_lines = []

        for entry in recognized_lines:

            line_id = entry["line_id"]

            if line_id >= len(line_crops):
                continue

            corrected = (
                corrected_lines[line_id]
                if line_id < len(corrected_lines)
                else entry["text"]
            )

            crop = line_crops[line_id]["image"]

            if self._verify_line(crop, corrected):
                validated_lines.append(corrected)
            else:
                validated_lines.append(entry["text"])

        return "\n".join(validated_lines)

    def _verify_line(self, image, text):

        pil_img = self._to_pil(image)

        prompt = f"""
    <image>

    The image contains handwritten text.

    Does the following transcription exactly match the text in the image?

    Text:
    {text}

    Answer ONLY YES or NO.
    """

        inputs = self.processor(
            images=pil_img,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5
            )

        response = self.processor.decode(
            outputs[0],
            skip_special_tokens=True
        ).lower()

        return "yes" in response

    def _to_pil(self, img):

        if len(img.shape) == 2:
            return Image.fromarray(img).convert("RGB")

        return Image.fromarray(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )