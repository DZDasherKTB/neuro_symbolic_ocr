import numpy as np
from typing import List, Dict


class LineBuilder:

    def __init__(self, vertical_threshold: float = 0.6):
        """
        vertical_threshold controls how tolerant line grouping is.
        Higher = more tolerant to slanted handwriting.
        """
        self.vertical_threshold = vertical_threshold

    def assign_line_ids(self, detections: List[Dict]) -> List[Dict]:

        if not detections:
            return []

        detections = sorted(detections, key=lambda d: d["bbox"][1])

        lines = []
        line_id = 0

        for det in detections:

            x1, y1, x2, y2 = det["bbox"]
            center_y = (y1 + y2) / 2
            height = y2 - y1

            assigned = False

            for line in lines:

                if abs(center_y - line["center_y"]) < height * self.vertical_threshold:

                    det["line_id"] = line["id"]
                    line["boxes"].append(det)
                    assigned = True
                    break

            if not assigned:

                det["line_id"] = line_id

                lines.append({
                    "id": line_id,
                    "center_y": center_y,
                    "boxes": [det]
                })

                line_id += 1

        return detections
    
    