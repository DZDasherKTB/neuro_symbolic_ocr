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

        model_path = hf_hub_download(
            local_dir=".",
            repo_id="armvectores/yolov8n_handwritten_text_detection",
            filename="best.pt"
        )

        self.model = YOLO(model_path)
        self.conf_thresh = self.config.box_thresh

    def detect(self, tiles: List[Dict[str, Any]]) -> Dict[str, Any]:

        detections = []

        for tile in tiles:

            tile_img = tile["image"]
            tile_id = tile["tile_id"]
            x_offset, y_offset, _, _ = tile["global_coords"]

            if len(tile_img.shape) == 2:
                tile_img = cv2.cvtColor(tile_img, cv2.COLOR_GRAY2BGR)

            results = self.model.predict(
                tile_img,
                conf=self.conf_thresh,
                imgsz=960,
                iou=0.5,
                device=self.device,
                verbose=False
            )

            for r in results:

                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()

                for box, score in zip(boxes, scores):

                    x1, y1, x2, y2 = box.astype(int)

                    detections.append({
                        "bbox": [
                            int(x1 + x_offset),
                            int(y1 + y_offset),
                            int(x2 + x_offset),
                            int(y2 + y_offset)
                        ],
                        "confidence": float(score),
                        "tile_id": tile_id
                    })

        detections = self._apply_nms(detections)
        detections = self._merge_overlapping_boxes(detections)
        detections = self._merge_horizontally(detections)
        detections = [
            d for d in detections
            if (d["bbox"][2]-d["bbox"][0]) > 40
            and (d["bbox"][3]-d["bbox"][1]) > 15
        ]
        
        return detections
        # lines = self._group_into_lines(detections)

        # paragraphs = self._group_into_paragraphs(lines)

        # return {
        #     "paragraphs": paragraphs
        # }

    def _apply_nms(self, detections):

        if not detections:
            return []

        boxes = [d["bbox"] for d in detections]
        scores = [d["confidence"] for d in detections]

        boxes_xywh = [
            [b[0], b[1], b[2]-b[0], b[3]-b[1]]
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

    def detect_text_regions(self, detections):

        if not detections:
            return []

        # sort boxes top → bottom
        detections = sorted(detections, key=lambda d: d["bbox"][1])

        regions = []
        current = [detections[0]]

        for i in range(1, len(detections)):

            prev = current[-1]
            curr = detections[i]

            prev_bottom = prev["bbox"][3]
            curr_top = curr["bbox"][1]

            height = prev["bbox"][3] - prev["bbox"][1]

            # if vertical gap small → same block
            if curr_top - prev_bottom < height * 2.5:
                current.append(curr)

            else:
                regions.append(current)
                current = [curr]

        regions.append(current)

        # convert groups → bounding boxes
        blocks = []

        for group in regions:

            x1 = min(d["bbox"][0] for d in group)
            y1 = min(d["bbox"][1] for d in group)
            x2 = max(d["bbox"][2] for d in group)
            y2 = max(d["bbox"][3] for d in group)

            pad = 20

            blocks.append([
                max(0, x1 - pad),
                max(0, y1 - pad),
                x2 + pad,
                y2 + pad
            ])

        return blocks
    
    # def _group_into_lines(self, detections):

    #     detections = sorted(detections, key=lambda d: d["bbox"][1])

    #     lines = []
    #     vertical_threshold = 0.6

    #     for det in detections:

    #         x1, y1, x2, y2 = det["bbox"]
    #         center_y = (y1 + y2) / 2
    #         height = y2 - y1

    #         assigned = False

    #         for line in lines:
    #             line_center = np.mean([
    #                 (b["bbox"][1] + b["bbox"][3]) / 2
    #                 for b in line["boxes"]
    #             ])

    #             if abs(center_y - line_center) < height * vertical_threshold:
    #                 line["boxes"].append(det)
    #                 assigned = True
    #                 break

    #         if not assigned:

    #             lines.append({
    #                 "center_y": center_y,
    #                 "boxes": [det]
    #             })

    #     for line in lines:
    #         line["boxes"] = sorted(line["boxes"], key=lambda d: d["bbox"][0])

    #     return lines

    # def _group_into_paragraphs(self, lines):

    #     if not lines:
    #         return []

    #     paragraphs = []
    #     line_heights=[max(b["bbox"][3]-b["bbox"][1] for b in l["boxes"]) for l in lines]
    #     avg_h=sum(line_heights)/len(line_heights)

    #     vertical_gap_threshold = avg_h*2

    #     current_para = [lines[0]]

    #     for i in range(1, len(lines)):

    #         prev_line = lines[i-1]
    #         curr_line = lines[i]

    #         prev_bottom = max(b["bbox"][3] for b in prev_line["boxes"])
    #         curr_top = min(b["bbox"][1] for b in curr_line["boxes"])

    #         if curr_top - prev_bottom < vertical_gap_threshold:

    #             current_para.append(curr_line)

    #         else:

    #             paragraphs.append(current_para)
    #             current_para = [curr_line]

    #     paragraphs.append(current_para)

    #     return paragraphs
    
    def _merge_overlapping_boxes(self, detections, iou_thresh=0.5):

        changed = True

        while changed:

            changed = False
            new_detections = []

            used = [False] * len(detections)

            for i in range(len(detections)):

                if used[i]:
                    continue

                boxA = detections[i]["bbox"]

                for j in range(i+1, len(detections)):

                    if used[j]:
                        continue

                    boxB = detections[j]["bbox"]

                    if self._iou(boxA, boxB) > iou_thresh:

                        boxA = [
                            min(boxA[0], boxB[0]),
                            min(boxA[1], boxB[1]),
                            max(boxA[2], boxB[2]),
                            max(boxA[3], boxB[3]),
                        ]

                        used[j] = True
                        changed = True

                new_detections.append({
                    "bbox": boxA,
                    "confidence": detections[i]["confidence"],
                    "tile_id": detections[i]["tile_id"]
                })

                used[i] = True

            detections = new_detections

        return detections
    
    def _iou(self, a, b):

        xA = max(a[0], b[0])
        yA = max(a[1], b[1])
        xB = min(a[2], b[2])
        yB = min(a[3], b[3])

        inter = max(0, xB-xA) * max(0, yB-yA)

        areaA = (a[2]-a[0])*(a[3]-a[1])
        areaB = (b[2]-b[0])*(b[3]-b[1])

        return inter / (areaA + areaB - inter + 1e-6)
    
    def _merge_horizontally(self, detections, gap_ratio=1.5):

        detections = sorted(detections, key=lambda d: d["bbox"][0])

        merged = []
        used = [False]*len(detections)

        for i in range(len(detections)):

            if used[i]:
                continue

            x1,y1,x2,y2 = detections[i]["bbox"]
            h = y2-y1

            for j in range(i+1,len(detections)):

                if used[j]:
                    continue

                bx1,by1,bx2,by2 = detections[j]["bbox"]

                center_diff = abs(((y1+y2)/2)-((by1+by2)/2))

                # same line
                if center_diff < h*0.5:

                    gap = bx1-x2

                    if gap < h*gap_ratio:

                        x2 = max(x2,bx2)
                        y1 = min(y1,by1)
                        y2 = max(y2,by2)

                        used[j]=True

            merged.append({
                "bbox":[x1,y1,x2,y2],
                "confidence":detections[i]["confidence"],
                "tile_id":detections[i]["tile_id"]
            })

            used[i]=True

        return merged