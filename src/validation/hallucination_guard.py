import numpy as np
from typing import List, Dict, Any


class HallucinationGuard:
    """
    Final arbitration layer combining OCR, LLM reasoning, and VLM visual verification.
    """

    def __init__(self):
        # Fusion weights (can later move to config)
        self.w_ocr = 0.4
        self.w_vlm = 0.4
        self.w_llm = 0.2

    @staticmethod
    def calculate_fusion_score(ocr_prob: float, vlm_match: bool, llm_weight: float) -> float:
        """
        Combines probabilities from OCR, VLM validation, and LLM reasoning.
        """
        vlm_score = 1.0 if vlm_match else 0.0
        return (ocr_prob * 0.4) + (vlm_score * 0.4) + (llm_weight * 0.2)

    def resolve(
        self,
        corrected_lines,
        recognized_lines,
        line_crops,
        vlm_validator
    ):

        final_lines = []

        for i, corrected in enumerate(corrected_lines):

            if i >= len(line_crops):
                break

            crop = line_crops[i]["image"]

            vlm_match = vlm_validator._verify_line(crop, corrected)

            if vlm_match:
                final_lines.append(corrected)
            else:
                final_lines.append(recognized_lines[i]["text"])

        return "\n".join(final_lines)