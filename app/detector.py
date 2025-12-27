from ultralytics import YOLO

class ModelRegistry:
    def __init__(self):
        self.models = {
            "yolov8n": YOLO("yolov8n.pt"),
            "yolov8s": YOLO("yolov8s.pt")
        }

    def detect(self, image, model_name="yolov8n"):
        """
        Runs YOLO detection and returns bounding boxes + metadata.
        NO drawing happens here.
        """
        model = self.models.get(model_name, self.models["yolov8n"])
        results = model(image)

        yolo_boxes = []
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])

                yolo_boxes.append((x1, y1, x2, y2, label, conf))
                detections.append({
                    "label": label,
                    "confidence": round(conf, 3)
                })

        return yolo_boxes, detections
