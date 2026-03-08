import torch
import numpy as np
import cv2
from typing import List, Dict, Any
from paddleocr import PaddleOCR
from src.pipeline.config_loader import ConfigLoader

class DBNetDetector:
    def __init__(self):
        config = ConfigLoader()
        self.config = config.models.detection
        self.device = config.pipeline.device
        
        self.box_thresh = self.config.box_thresh 
        
        print("Loading PaddleOCR DBNet detector...")
        self.ocr_detector = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            det=True,
            rec=False,
            use_gpu=(self.device == "cuda"),
            det_db_thresh=0.3, 
            det_db_box_thresh=self.box_thresh 
        )

    def detect(self, tiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        global_detections = []

        for tile in tiles:
            tile_img = tile["image"]
            tile_id = tile["tile_id"]
            x_offset, y_offset, _, _ = tile["global_coords"]

            results = self.ocr_detector.ocr(tile_img, det=True, rec=False, cls=False)

            if not results or results[0] is None:
                continue

            for line in results:
                for det in line:
                    poly_pts = np.array(det[0]).astype(int)
                    confidence = float(det[1])

                    x1, y1 = poly_pts[:, 0].min(), poly_pts[:, 1].min()
                    x2, y2 = poly_pts[:, 0].max(), poly_pts[:, 1].max()

                    global_box = self._to_global([x1, y1, x2, y2], x_offset, y_offset)

                    global_detections.append({
                        "parent_tile_id": tile_id,
                        "bbox": global_box,
                        "confidence": confidence,
                        "style_tag": None  #needs refinement (later)
                    })

        return self._apply_nms(global_detections)

    def _to_global(self, box: List[int], x_off: int, y_off: int) -> List[int]:
        return [
            int(box[0] + x_off),
            int(box[1] + y_off),
            int(box[2] + x_off),
            int(box[3] + y_off)
        ]

    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        if not detections:
            return []

        boxes = [d["bbox"] for d in detections]
        scores = [d["confidence"] for d in detections]

        # Convert [x1, y1, x2, y2] to [x, y, w, h] for cv2 NMS
        boxes_xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            scores,
            score_threshold=0.2,
            nms_threshold=0.4     # Overlap threshold
        )

        if len(indices) == 0:
            return []

        return [detections[i] for i in indices.flatten()]