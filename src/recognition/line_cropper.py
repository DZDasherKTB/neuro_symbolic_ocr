class LineCropper:

    @staticmethod
    def crop_lines(image, detections):

        lines = {}

        for det in detections:

            line_id = det["line_id"]

            if line_id not in lines:
                lines[line_id] = []

            lines[line_id].append(det["bbox"])

        line_crops = []

        for line_id, boxes in lines.items():

            x1 = min(b[0] for b in boxes)
            y1 = min(b[1] for b in boxes)
            x2 = max(b[2] for b in boxes)
            y2 = max(b[3] for b in boxes)

            crop = image[y1:y2, x1:x2]

            line_crops.append({
                "line_id": line_id,
                "image": crop
            })

        return line_crops