import torch
from PIL import Image
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.pipeline.config_loader import ConfigLoader

class DocumentProfile:
    """Data class to hold VLM-derived document characteristics."""
    def __init__(self, script_type: str, condition: str, has_noise: bool, metadata: Dict):
        self.script_type = script_type  # cursive, block, mixed
        self.condition = condition      # faded, clean, stained
        self.has_noise = has_noise      # scanner artifacts, bleed-through
        self.metadata = metadata

class DocumentProfiler:
    def __init__(self):
        self.config = ConfigLoader().preprocessing.profiler
        self.device = ConfigLoader().pipeline.device
        
        print(f"Loading Document Profiler: {self.config.model}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model, 
            trust_remote_code=True,
            device_map=self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)

    def profile(self, image_path: str) -> DocumentProfile:
        """
        Analyzes the image to determine optimal OCR strategies.
        """
        image = Image.open(image_path).convert("RGB")
        
        prompt = (
            "Describe this handwritten document in terms of: "
            "1. Script type (cursive or block) "
            "2. Condition (clean or faded) "
            "3. Background noise (low or high). "
            "Provide answer in format: Script: [Type], Condition: [Cond], Noise: [Level]"
        )
        
        with torch.no_grad():
            analysis = self.model.answer_question(
                self.model.encode_image(image), 
                prompt, 
                self.tokenizer
            )
        
        return self._parse_analysis(analysis)

    def _parse_analysis(self, text: str) -> DocumentProfile:
        """Parses VLM output into a structured DocumentProfile object."""
        text = text.lower()
        
        # Simple heuristic parsing (can be replaced with regex or small LLM later)
        script = "cursive" if "cursive" in text else "block"
        condition = "faded" if "faded" in text else "clean"
        noise = "high" in text or "noisy" in text
        
        return DocumentProfile(
            script_type=script,
            condition=condition,
            has_noise=noise,
            metadata={"raw_vlm_output": text}
        )