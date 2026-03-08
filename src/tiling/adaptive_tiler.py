import numpy as np
import cv2
from typing import List, Tuple, Dict, Any
from src.pipeline.config_loader import ConfigLoader

class AdaptiveTiler:
    def __init__(self):
        self.config = ConfigLoader().tiling.tiling_engine
        self.scales = self.config.scales
        self.overlap_ratio = self.config.overlap_ratio

    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates multi-scale tiles from the normalized image.
        Returns a list of dicts containing the tile image and its global coordinates.
        """
        all_tiles = []
        h, w = image.shape[:2]

        for scale in self.scales:
            tile_h, tile_w = scale['resolution']
            stride = scale['stride']

            for y in range(0, h - tile_h + 1, stride):
                for x in range(0, w - tile_w + 1, stride):
                    tile = image[y:y+tile_h, x:x+tile_w]
                    
                    if self._is_informative(tile):
                        all_tiles.append({
                            'image': tile,
                            'coords': (x, y, x + tile_w, y + tile_h),
                            'scale': tile_h
                        })
            all_tiles.extend(self._get_edge_tiles(image, scale))

        return all_tiles

    def _is_informative(self, tile: np.ndarray, threshold: float = 0.01) -> bool:
        """
        Determines if a tile contains enough ink to be worth processing.
        Uses pixel density as a proxy for information.
        """
        ink_pixels = np.sum(tile < 250) 
        total_pixels = tile.size
        return (ink_pixels / total_pixels) > threshold

    def _get_edge_tiles(self, image: np.ndarray, scale: Dict) -> List[Dict]:
        """Captures the remaining pixels on the right and bottom boundaries."""
        h, w = image.shape[:2]
        tile_h, tile_w = scale['resolution']
        edge_tiles = []

        for x in range(0, w - tile_w + 1, scale['stride']):
            y = h - tile_h
            edge_tiles.append({
                'image': image[y:h, x:x+tile_w],
                'coords': (x, y, x + tile_w, h),
                'scale': tile_h
            })
            
        for y in range(0, h - tile_h + 1, scale['stride']):
            x = w - tile_w
            edge_tiles.append({
                'image': image[y:y+tile_h, x:w],
                'coords': (x, y, w, y + tile_h),
                'scale': tile_h
            })
            
        return edge_tiles