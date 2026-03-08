import numpy as np
from typing import List, Dict, Any
from src.pipeline.config_loader import ConfigLoader

class LatticeBuilder:
    def __init__(self):
        self.config = ConfigLoader().models.recognition.lattice
        self.line_threshold = 0.5 
    def build_structured_lattice(self, recognition_results: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Groups word-level candidates into a line-by-line structure.
        Returns: List of Lines, where each Line is a List of Word Objects (with candidates).
        """
        if not recognition_results:
            return []

        recognition_results.sort(key=lambda x: x['bbox'][1])

        lines = []
        if recognition_results:
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
        """
        Converts the grouped lines into a prompt-friendly format.
        """
        formatted_lattice = []
        for line_idx, line in enumerate(lines):
            line_data = []
            for word in line:
                line_data.append({
                    "word_pos": len(line_data),
                    "candidates": word['candidates'],
                    "visual_conf": round(word['confidence_ocr'], 3)
                })
            formatted_lattice.append({
                "line_id": line_idx,
                "words": line_data
            })
        return formatted_lattice