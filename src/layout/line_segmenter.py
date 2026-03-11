import cv2
import numpy as np


class LineSegmenter:

    def segment_lines(self, image):

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        projection = np.sum(binary, axis=1)

        projection = projection / np.max(projection)

        threshold = 0.15

        lines = []
        start = None

        for i, val in enumerate(projection):

            if val > threshold and start is None:
                start = i

            elif val <= threshold and start is not None:

                if i - start > 10:
                    lines.append((start, i))

                start = None

        if start is not None:
            lines.append((start, len(projection)))

        return lines
      
    def extract_line_crops(self, image):

        segments = self.segment_lines(image)

        line_crops = []

        for i, (y1, y2) in enumerate(segments):

            pad = 8

            y1 = max(0, y1 - pad)
            y2 = min(image.shape[0], y2 + pad)

            crop = image[y1:y2, :]

            if crop.shape[0] < 10:
                continue

            line_crops.append({
                "line_id": i,
                "bbox": [0, y1, image.shape[1], y2],
                "image": crop
            })

        return line_crops
      