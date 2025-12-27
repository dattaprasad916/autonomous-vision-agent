import cv2
import time

from app.detector import ModelRegistry
from cognition.embedding import extract_embedding
from cognition.persistence import PersistenceEngine

# --------------------------------------------------
# INITIAL SETUP
# --------------------------------------------------
model_registry = ModelRegistry()
memory_engine = PersistenceEngine()

current_model = "yolov8n"

# Adaptive confidence control
confidence_threshold = 0.4
MIN_THRESHOLD = 0.3
MAX_THRESHOLD = 0.6

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Could not open webcam")
    exit()

print("\n🤖 Autonomous Vision Agent Started")
print("System is self-regulating (no user control required)")
print("Press [q] to quit\n")

prev_time = 0

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))

    # ---------------------------
    # PERCEPTION
    # ---------------------------
    yolo_boxes, _ = model_registry.detect(frame, current_model)

    new_count = 0
    total_count = len(yolo_boxes)

    # ---------------------------
    # COGNITION + FILTERING
    # ---------------------------
    for x1, y1, x2, y2, label, conf in yolo_boxes:

        # Autonomous confidence gating
        if conf < confidence_threshold:
            continue

        bbox = (x1, y1, x2, y2)
        embedding = extract_embedding(frame, bbox)
        if embedding is None:
            continue

        obj, status = memory_engine.match_or_create(embedding)

        if status == "new":
            new_count += 1
            color = (0, 0, 255)     # RED = NEW
            tag = "NEW"
        else:
            color = (0, 255, 0)     # GREEN = KNOWN
            tag = "KNOWN"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{tag} | seen {obj.seen_count}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # ---------------------------
    # AUTONOMOUS DECISION
    # ---------------------------
    if total_count > 0:
        novelty_ratio = new_count / total_count

        # Self-regulating confidence threshold
        if novelty_ratio > 0.4:
            confidence_threshold = max(
                MIN_THRESHOLD, confidence_threshold - 0.01
            )
        elif novelty_ratio < 0.2:
            confidence_threshold = min(
                MAX_THRESHOLD, confidence_threshold + 0.01
            )

    # ---------------------------
    # PERFORMANCE
    # ---------------------------
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if prev_time else 0
    prev_time = curr_time

    # ---------------------------
    # UI OVERLAY
    # ---------------------------
    cv2.rectangle(frame, (0, 0), (960, 120), (20, 20, 20), -1)

    cv2.putText(frame, f"Model: {current_model}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f"FPS: {fps}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f"Confidence threshold: {confidence_threshold:.2f}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

    cv2.putText(frame, f"Memory size: {len(memory_engine.memory_bank)}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

    cv2.putText(frame, "RED: New | GREEN: Known",
                (400, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Autonomous Vision Agent — Live", frame)

    # ---------------------------
    # EXIT
    # ---------------------------
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --------------------------------------------------
# CLEANUP
# --------------------------------------------------
memory_engine.save()
cap.release()
cv2.destroyAllWindows()

