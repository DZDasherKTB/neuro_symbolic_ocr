import numpy as np
from typing import Dict, Any, List


class LineCropper:

    @staticmethod
    def crop_lines(image: np.ndarray, layout: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Crops lines from the structured detector output.

        layout format:
        {
            "paragraphs": [
                [line1, line2],
                [line3]
            ]
        }
        """

        line_crops = []

        paragraphs = layout.get("paragraphs", [])

        for para_id, paragraph in enumerate(paragraphs):

            for line_id, line in enumerate(paragraph):

                boxes = [b["bbox"] for b in line["boxes"]]

                x1 = min(b[0] for b in boxes)
                y1 = min(b[1] for b in boxes)
                x2 = max(b[2] for b in boxes)
                y2 = max(b[3] for b in boxes)

                crop = image[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                line_crops.append({
                    "paragraph_id": para_id,
                    "line_id": line_id,
                    "bbox": [x1, y1, x2, y2],
                    "image": crop,
                    "num_words": len(boxes)
                })

        return line_crops