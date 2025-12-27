import cv2

def discover_objects(image):
    """
    Generic object discovery using OpenCV.
    Finds visually distinct regions without labeling.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 1500:  # ignore noise
            boxes.append((x, y, x + w, y + h))

    return boxes
