import torch
import cv2
from PIL import Image
from typing import List, Dict, Any
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from src.pipeline.config_loader import ConfigLoader

class VLMValidator:
    def __init__(self):
        config = ConfigLoader()
        self.cfg = config.models.validation
        self.device = config.pipeline.device
        
        print(f"Loading Hallucination Guard: {self.cfg.model_name}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.cfg.model_name, 
            torch_dtype=torch.bfloat16, 
            device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(self.cfg.model_name)
        self.threshold = self.cfg.cross_modal_threshold # e.g., 0.75

    def validate(self, proposed_text: str, visual_evidence: List[Dict], lattice: List[Dict]) -> str:
        """
        Final check: Ensures LLM corrections are visually grounded.
        """
        proposed_words = proposed_text.split()
        final_output = []

        for i, (word_obj, lattice_entry) in enumerate(zip(proposed_words, lattice)):
            ocr_top_1 = lattice_entry['candidates'][0]['text']
            
            if word_obj.lower() != ocr_top_1.lower():
                crop = visual_evidence[i]['image']
                is_valid = self._verify_with_vlm(crop, word_obj)
                
                if is_valid:
                    final_output.append(word_obj) 
                else:
                    final_output.append(ocr_top_1)
            else:
                final_output.append(word_obj)

        return " ".join(final_output)

    def _verify_with_vlm(self, crop: np.ndarray, word: str) -> bool:
        """
        Asks Qwen2-VL: 'Does this image snippet actually contain the word X?'
        """
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": f"Question: Does the word '{word}' appear in this image? Answer ONLY 'Yes' or 'No'."},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil_img], padding=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=10)
            
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return "yes" in response.lower()