import torch
import json
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.pipeline.config_loader import ConfigLoader

class LLMReconstructor:
    def __init__(self):
        config = ConfigLoader()
        self.cfg = config.models.reasoning
        self.device = config.pipeline.device
        
        print(f"Loading Reasoning Engine: {self.cfg.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            # load_in_8bit=True # Optional
        )
        self.model.eval()
        self.lambda_visual = self.cfg.optimization.lambda_visual # 0.65

    def reconstruct(self, lattice: List[Dict], context: str = "") -> str:
        """
        Feeds the word lattice to Llama-3 to find the most probable sentence.
        """
        prompt = self._build_prompt(lattice, context)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                do_sample=True
            )
        
        response = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return self._extract_final_text(response, prompt)

    def _build_prompt(self, lattice: List[Dict], context: str) -> str:
        """
        Creates a structured prompt containing the token lattice.
        """
        lattice_str = json.dumps(lattice, indent=2)
        
        system_msg = (
            "You are a handwriting expert. You will receive a 'Token Lattice' representing "
            "OCR candidates for a handwritten document. Each word has multiple candidates "
            "with visual probabilities. Your task is to reconstruct the most likely original "
            "text based on grammar, context, and the provided probabilities.\n\n"
            f"Context of document: {context}\n"
            "Output ONLY the final reconstructed text."
        )
        
        user_msg = f"Lattice Data:\n{lattice_str}\n\nReconstructed Text:"
        
        # Using Llama-3 Chat Template format
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>" \
               f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>" \
               f"<|start_header_id|>assistant<|end_header_id|>\n\n"

    def _extract_final_text(self, response: str, prompt: str) -> str:
        """Removes the prompt part of the response."""
        return response[len(prompt):].strip()