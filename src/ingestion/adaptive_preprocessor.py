import cv2
import numpy as np
from PIL import Image
from skimage.filters import threshold_sauvola
from deskew import determine_skew
from src.pipeline.config_loader import ConfigLoader
from typing import Any
class AdaptivePreprocessor:
    def __init__(self):
        self.config = ConfigLoader().preprocessing
        
    def process(self, image_path: str, profile: Any) -> np.ndarray:
        """
        Main entry point: Orchestrates preprocessing based on the document profile.
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if profile.has_noise:
            img = self.apply_median_filter(img, self.config.augmentations.blur.median_kernel)

        img = self.deskew(img)

        if profile.condition == "faded":
            window_size = self.config.thresholding.window_size + 10 
        else:
            window_size = self.config.thresholding.window_size
            
        img = self.apply_sauvola_threshold(img, window_size)

        return img

    def apply_median_filter(self, img: np.ndarray, kernel_size: int) -> np.ndarray:
        """Removes salt-and-pepper noise often found in scans."""
        return cv2.medianBlur(img, kernel_size)

    def apply_sauvola_threshold(self, img: np.ndarray, window_size: int) -> np.ndarray:
        """
        Sauvola thresholding is superior for documents with uneven lighting
        or staining (common in historical sources).
        """
        thresh_sauvola = threshold_sauvola(img, window_size=window_size, k=self.config.thresholding.k_factor)
        binary_sauvola = img > thresh_sauvola
        return (binary_sauvola * 255).astype(np.uint8)

    def deskew(self, img: np.ndarray) -> np.ndarray:
        """Corrects the tilt of the document to improve text line detection."""
        angle = determine_skew(img)
        if angle is None or abs(angle) < 0.5:
            return img
            
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def grayscale(self, img: np.ndarray) -> np.ndarray:
        """Standard grayscale conversion if not already handled."""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img