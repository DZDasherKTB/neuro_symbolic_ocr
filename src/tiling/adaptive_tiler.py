import numpy as np
import cv2
import uuid
from typing import List, Tuple, Dict, Any
from src.pipeline.config_loader import ConfigLoader

class AdaptiveTiler:
    def __init__(self):
        self.config = ConfigLoader().tiling.tiling_engine
        self.scales = self.config.scales  
        self.overlap_ratio = self.config.overlap_ratio

    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates multi-scale tiles with unique IDs for tracking.
        Returns: List of dicts {id, image, coords, scale}
        """
        all_tiles = []
        h, w = image.shape[:2]

        for scale in self.scales:
            tile_h, tile_w = scale['resolution']
            stride = scale['stride']

            for y in range(0, h - tile_h + 1, stride):
                for x in range(0, w - tile_w + 1, stride):
                    tile_img = image[y:y+tile_h, x:x+tile_w]
                    
                    if self._is_informative(tile_img):
                        all_tiles.append(self._create_tile_entry(tile_img, x, y, tile_w, tile_h))

            all_tiles.extend(self._get_edge_tiles(image, scale))

        return all_tiles

    def _create_tile_entry(self, img: np.ndarray, x: int, y: int, tw: int, th: int) -> Dict[str, Any]:
        """Creates a structured tile object with a unique identifier."""
        return {
            "tile_id": str(uuid.uuid4())[:8],
            "image": img,
            "global_coords": (x, y, x + tw, y + th), # (x1, y1, x2, y2)
            "resolution": (tw, th)
        }

    def _is_informative(self, tile: np.ndarray, threshold: float = 0.01) -> bool:
        """Heuristic to skip empty tiles by checking ink density."""
        ink_pixels = np.sum(tile < 250) 
        return (ink_pixels / tile.size) > threshold

    def _get_edge_tiles(self, image: np.ndarray, scale: Dict[str, Any]) -> List[Dict[str, Any]]:
      """
      Adds tiles for the right edge, bottom edge, and bottom-right corner
      if the sliding window did not already cover them.
      """
      h, w = image.shape[:2]
      tile_h, tile_w = scale["resolution"]
      stride = scale["stride"]

      edge_tiles = []

      last_x = (w - tile_w) // stride * stride
      last_y = (h - tile_h) // stride * stride

      if last_x + tile_w < w:
          x_start = w - tile_w
          for y in range(0, h - tile_h + 1, stride):
              tile = image[y:y+tile_h, x_start:x_start+tile_w].copy()
              if self._is_informative(tile):
                  edge_tiles.append(
                      self._create_tile_entry(tile, x_start, y, tile_w, tile_h)
                  )

      if last_y + tile_h < h:
          y_start = h - tile_h
          for x in range(0, w - tile_w + 1, stride):
              tile = image[y_start:y_start+tile_h, x:x+tile_w].copy()
              if self._is_informative(tile):
                  edge_tiles.append(
                      self._create_tile_entry(tile, x, y_start, tile_w, tile_h)
                  )

      if (last_x + tile_w < w) and (last_y + tile_h < h):
          x_start = w - tile_w
          y_start = h - tile_h

          tile = image[y_start:y_start+tile_h, x_start:x_start+tile_w].copy()

          if self._is_informative(tile):
              edge_tiles.append(
                  self._create_tile_entry(tile, x_start, y_start, tile_w, tile_h)
              )

      return edge_tiles