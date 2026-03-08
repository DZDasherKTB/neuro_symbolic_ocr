import torch
from PIL import Image
from typing import List, Dict, Any
from src.pipeline.config_loader import ConfigLoader
from src.detection.bbox_utils import BBoxUtils
import cv2
import numpy as np
class BBoxRefiner:
    def __init__(self, vlm_model=None):
        config = ConfigLoader()
        self.config = config.models.detection.vlm_refiner
        self.trigger_threshold = self.config.trigger_threshold # e.g., 0.4
        self.vlm = vlm_model 
        
    def refine(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Filters detections using VLM verification for 'marginal' boxes.
        """
        refined_detections = []
        
        for det in detections:
            conf = det['confidence']
            
            # 1. High confidence: Keep as is
            if conf >= 0.7:
                refined_detections.append(det)
                
            elif self.trigger_threshold <= conf < 0.7:
                crop = BBoxUtils.crop_with_padding(image, det['bbox'])
                if self._vlm_confirm_text(crop):
                    det['confidence'] = 0.75
                    refined_detections.append(det)
                    
            else:
                continue
                
        return refined_detections

    def _vlm_confirm_text(self, crop_img: np.ndarray) -> bool:
        """
        Asks VLM: 'Is there actual text in this image?'
        """
        if self.vlm is None:
            return True 
            
        if len(crop_img.shape) == 2:
            pil_img = Image.fromarray(crop_img).convert("RGB")
        else:
            pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))        
        
        prompt = "Does this image contain any handwritten or printed text? Answer yes or no."
                
        response = self.vlm.generate_response(pil_img, prompt).lower()
        return "yes" in response