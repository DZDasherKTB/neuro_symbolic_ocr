import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.pipeline.config_loader import ConfigLoader


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

    # -----------------------------------------------------

    def reconstruct(self, recognized_lines: List[Dict]) -> str:
        """
        Corrects OCR output and reconstructs readable text.
        Supports multiple text blocks.
        """

        if not recognized_lines:
            return ""

        # sort by block then line
        recognized_lines = sorted(
            recognized_lines,
            key=lambda x: (x.get("block_id", 0), x["line_id"])
        )

        blocks = {}

        for line in recognized_lines:
            block_id = line.get("block_id", 0)

            if block_id not in blocks:
                blocks[block_id] = []

            blocks[block_id].append(line["text"])

        raw_blocks = []

        for block_id in sorted(blocks.keys()):
            raw_blocks.append("\n".join(blocks[block_id]))

        raw_text = "\n\n".join(raw_blocks)

        prompt = self._build_prompt(raw_text)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():

            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )

        response = self.tokenizer.decode(
            output_tokens[0],
            skip_special_tokens=True
        )

        return self._extract_final_text(response, prompt)

    # -----------------------------------------------------

    def _build_prompt(self, raw_text: str) -> str:

        prompt = f"""
You are correcting OCR output from handwritten text.

The OCR system already detected lines correctly but may contain spelling mistakes.

Rules:
- Fix spelling errors
- Fix punctuation
- Preserve original meaning
- Do NOT invent new sentences
- Keep paragraph structure

OCR text:
{raw_text}

Corrected text:
"""

        print("\n===== LLM PROMPT START =====\n")
        print(prompt)
        print("\n===== LLM PROMPT END =====\n")

        return prompt

    # -----------------------------------------------------

    def _extract_final_text(self, response: str, prompt: str) -> str:

        return response[len(prompt):].strip()