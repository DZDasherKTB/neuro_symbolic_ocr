import cv2
import numpy as np
from src.pipeline.config_loader import ConfigLoader

class StrokeNormalizer:
    def __init__(self):
        self.config = ConfigLoader().preprocessing.normalization
        self.target_width = self.config.target_stroke_width # Usually 10px

    def normalize(self, binary_img: np.ndarray) -> np.ndarray:
        """
        Estimates the current stroke width and rescales the image to reach the target width.
        Input: Binarized image (from AdaptivePreprocessor).
        """
        current_width = self.estimate_stroke_width(binary_img)
        
        if current_width == 0:
            return binary_img
            
        scale_factor = self.target_width / current_width
        
        if 0.9 < scale_factor < 1.1:
            return binary_img

        height, width = binary_img.shape
        new_size = (int(width * scale_factor), int(height * scale_factor))
        
        normalized_img = cv2.resize(
            binary_img, 
            new_size, 
            interpolation=cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_CUBIC
        )
        
        return normalized_img

    def estimate_stroke_width(self, binary_img: np.ndarray) -> float:
        """
        Calculates stroke width using the Distance Transform method.
        The distance transform calculates the distance to the nearest background pixel.
        The local maxima in the skeleton represent half the stroke width.
        """
        if np.mean(binary_img) > 127:
            binary_img = cv2.bitwise_not(binary_img)

        dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
        
        _, skeleton = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)
        # Simple skeletonization (can be swapped for Zhang-Suen if needed)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        skeleton = cv2.ximgproc.thinning(binary_img)

        stroke_values = dist_transform[skeleton > 0]
        
        if len(stroke_values) == 0:
            return 0.0

        avg_radius = np.median(stroke_values) 
        return float(avg_radius * 2)