import numpy as np
from typing import List, Dict, Any
from src.pipeline.config_loader import ConfigLoader


class LatticeBuilder:

    def __init__(self):
        self.config = ConfigLoader().models.recognition.lattice
        self.line_threshold = 0.5

    def build_structured_lattice(self, recognition_results: List[Dict[str, Any]]) -> List[Dict]:

        if not recognition_results:
            return []

        # sort by vertical position
        recognition_results.sort(key=lambda x: x['bbox'][1])

        lines = []

        current_line = [recognition_results[0]]

        for i in range(1, len(recognition_results)):

            prev_bbox = current_line[-1]['bbox']
            curr_bbox = recognition_results[i]['bbox']

            prev_h = prev_bbox[3] - prev_bbox[1]
            overlap = min(prev_bbox[3], curr_bbox[3]) - max(prev_bbox[1], curr_bbox[1])

            if overlap > (prev_h * self.line_threshold):

                current_line.append(recognition_results[i])

            else:

                current_line.sort(key=lambda x: x['bbox'][0])
                lines.append(current_line)
                current_line = [recognition_results[i]]

        current_line.sort(key=lambda x: x['bbox'][0])
        lines.append(current_line)

        return self._format_for_llm(lines)

    def _format_for_llm(self, lines: List[List[Dict]]) -> List[Dict]:

        formatted_lattice = []

        for line_idx, line in enumerate(lines):

            words = []

            for word in line:

                candidates = word.get("candidates", [])

                clean_candidates = []

                for c in candidates:

                    if isinstance(c, dict):
                        clean_candidates.append(str(c.get("text", "")))
                    else:
                        clean_candidates.append(str(c))

                if clean_candidates:
                    words.append(clean_candidates)

            formatted_lattice.append({
                "line": line_idx,
                "words": words
            })

        return formatted_lattice