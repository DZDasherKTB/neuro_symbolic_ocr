import cv2
import numpy as np
from typing import List, Tuple

class BBoxUtils:
    @staticmethod
    def crop_with_padding(image: np.ndarray, bbox: List[int], padding: int = 5) -> np.ndarray:
        """
        Crops a bounding box from the image with extra padding.
        bbox: [x1, y1, x2, y2]
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(w, x2 + padding)
        y2_pad = min(h, y2 + padding)
        
        return image[y1_pad:y2_pad, x1_pad:x2_pad]

    @staticmethod
    def get_iou(boxA: List[int], boxB: List[int]) -> float:
        """Calculates Intersection over Union for two boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / float(boxAArea + boxBArea - interArea)