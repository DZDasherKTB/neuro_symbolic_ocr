import torch
from PIL import Image
from src.pipeline.config_loader import ConfigLoader
from transformers import AutoModelForVision2Seq, AutoProcessor


class DocumentProfile:
    def __init__(self, script: str, condition: str, noise: str):
        self.script = script
        self.condition = condition
        self.noise = noise

        self.has_noise = noise.lower() in ["medium", "high"]
        self.is_degraded = condition.lower() in ["degraded", "historical"]


class DocumentProfiler:

    def __init__(self):

        cfg = ConfigLoader().models.vlm

        self.model_id = cfg.model_name
        self.device = ConfigLoader().pipeline.device

        print(f"Loading Document Profiler VLM: {self.model_id}")

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )

        self.model.eval()

    def profile(self, image_path: str) -> DocumentProfile:

        image = Image.open(image_path).convert("RGB")

        prompt = """
            Look at the document image and classify it.

            Return EXACTLY three words separated by commas.

            Format:
            script, condition, noise

            Allowed values:
            script = cursive / printed / mixed
            condition = clean / degraded / historical
            noise = low / medium / high

            Example:
            cursive, degraded, medium
            """

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            if "pixel_attention_mask" in inputs:
                inputs.pop("pixel_attention_mask")

            output = self.model.generate(**inputs, max_new_tokens=30)

        response = self.processor.batch_decode(output, skip_special_tokens=True)[0]

        response = response.split("Assistant:")[-1].strip()

        response = response.split("\n")[0].strip()

        parts = [p.strip().lower() for p in response.split(",")]

        script = parts[0] if len(parts) > 0 else "mixed"
        condition = parts[1] if len(parts) > 1 else "clean"
        noise = parts[2] if len(parts) > 2 else "low"

        return DocumentProfile(script, condition, noise)