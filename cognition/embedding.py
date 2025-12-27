import cv2
import numpy as np

def extract_embedding(image, bbox):
    """
    Extracts a color-based embedding for an object crop.
    bbox = (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv], [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 180, 0, 256, 0, 256]
    )

    cv2.normalize(hist, hist)
    return hist.flatten()
