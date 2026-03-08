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
        proposed_text: str,
        visual_evidence: List[Dict[str, Any]],
        lattice: List[Dict[str, Any]],
        vlm_validator
    ) -> str:
        """
        Resolves disagreements between OCR and LLM using VLM confirmation.
        """

        proposed_words = proposed_text.split()
        final_words = []

        for i, (llm_word, lattice_entry) in enumerate(zip(proposed_words, lattice)):

            candidates = lattice_entry["candidates"]
            ocr_word = candidates[0]["text"]
            ocr_prob = candidates[0]["prob"]

            llm_prob = 1.0 if llm_word.lower() == ocr_word.lower() else 0.6

            crop = visual_evidence[i]["image"]

            vlm_match = vlm_validator._verify_with_vlm(crop, llm_word)

            fusion_score = self.calculate_fusion_score(
                ocr_prob=ocr_prob,
                vlm_match=vlm_match,
                llm_weight=llm_prob
            )

            if fusion_score >= 0.5:
                final_words.append(llm_word)
            else:
                final_words.append(ocr_word)

        return " ".join(final_words)