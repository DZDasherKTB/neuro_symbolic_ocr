import torch
import json
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.pipeline.config_loader import ConfigLoader
from transformers import BitsAndBytesConfig

class LLMReconstructor:
    def __init__(self):
        config = ConfigLoader()
        self.cfg = config.models.reasoning
        self.device = config.pipeline.device
        
        print(f"Loading Reasoning Engine: {self.cfg.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)

        bnb = BitsAndBytesConfig(
            load_in_4bit=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            quantization_config=bnb,
            device_map="auto"
        )
        self.model.eval()
        self.lambda_visual = self.cfg.optimization.lambda_visual # 0.65

    def reconstruct(self, lattice: List[Dict], context: str = "") -> str:
        """
        Feeds the word lattice to Llama-3 to find the most probable sentence.
        """
        prompt = self._build_prompt(lattice, context)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output_tokens = self.model.generate(
    **inputs,
    max_new_tokens=120,
    do_sample=False,
    temperature=None,
    top_p=None,
    top_k=None
)
        
        response = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return self._extract_final_text(response, prompt)

    def _build_prompt(self, lattice, context):

        lines = []

        for line in lattice:

            words = []

            for candidates in line["words"]:

                words.append(" / ".join(candidates))

            lines.append(" ".join(words))

        lattice_text = "\n".join(lines)

        prompt = f"""
    You are reconstructing handwritten OCR text.

    Each word may have multiple candidates separated by "/".

    Choose the most probable word from each group.

    Rules:
    - Do NOT invent words
    - Only choose from the candidates
    - Output ONLY the final reconstructed text

    OCR candidates:
    {lattice_text}

    Reconstructed text:
    """

        print("\n===== LLM PROMPT START =====\n")
        print(prompt)
        print("\n===== LLM PROMPT END =====\n")

        return prompt

    def _extract_final_text(self, response: str, prompt: str) -> str:
        """Removes the prompt part of the response."""
        return response[len(prompt):].strip()