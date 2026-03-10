import numpy as np
import cv2
from typing import List, Dict, Any
from ultralytics import YOLO
from src.pipeline.config_loader import ConfigLoader
from huggingface_hub import hf_hub_download


class DBNetDetector:

    def __init__(self):

        config = ConfigLoader()
        self.config = config.models.detection
        self.device = config.pipeline.device

        print("Loading YOLOv8 Text Detector...")


        model_path = hf_hub_download(local_dir=".",
                             repo_id="armvectores/yolov8n_handwritten_text_detection",
                             filename="best.pt")
        self.model = YOLO(model_path)

        self.conf_thresh = self.config.box_thresh

    def detect(self, tiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        global_detections = []

        for tile in tiles:

            tile_img = tile["image"]
            tile_id = tile["tile_id"]
            x_offset, y_offset, _, _ = tile["global_coords"]

            if len(tile_img.shape) == 2:
                tile_img = cv2.cvtColor(tile_img, cv2.COLOR_GRAY2BGR)

            results = self.model.predict(
                tile_img,
                conf=self.conf_thresh,
                imgsz=1280,
                iou=0.5,
                device=self.device,
                verbose=False
            )

            for r in results:

                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()

                for box, score in zip(boxes, scores):

                    x1, y1, x2, y2 = box.astype(int)

                    global_box = self._to_global(
                        [x1, y1, x2, y2],
                        x_offset,
                        y_offset
                    )

                    global_detections.append({
                        "parent_tile_id": tile_id,
                        "bbox": global_box,
                        "confidence": float(score),
                        "style_tag": None
                    })

        return self._apply_nms(global_detections)

    def _to_global(self, box, x_off, y_off):

        return [
            int(box[0] + x_off),
            int(box[1] + y_off),
            int(box[2] + x_off),
            int(box[3] + y_off)
        ]

    def _apply_nms(self, detections):

        if not detections:
            return []

        boxes = [d["bbox"] for d in detections]
        scores = [d["confidence"] for d in detections]

        boxes_xywh = [
            [b[0], b[1], b[2] - b[0], b[3] - b[1]]
            for b in boxes
        ]

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            scores,
            score_threshold=0.2,
            nms_threshold=0.4
        )

        if len(indices) == 0:
            return []

        return [detections[i] for i in indices.flatten()]