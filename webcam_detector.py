import cv2
from ultralytics import YOLO
import time


# ─── Config ───────────────────────────────────────────────
MODEL_SIZE    = 'yolov8s.pt'
CONFIDENCE    = 0.5
CAMERA_INDEX  = 0       # 0 = default webcam
SHOW_FPS      = True
# ──────────────────────────────────────────────────────────


def main():
    model = YOLO(MODEL_SIZE)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Webcam started. Press Q to quit.")

    prev_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read from webcam.")
            break

        # Run detection
        results   = model(frame, conf=CONFIDENCE, verbose=False)
        annotated = results[0].plot()

        # Show FPS
        if SHOW_FPS:
            curr_time = time.time()
            fps       = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(annotated, f'FPS: {int(fps)}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        # Show detection count
        count = len(results[0].boxes)
        cv2.putText(annotated, f'Objects: {count}',
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

        resized = cv2.resize(annotated, (1280, 720))
        cv2.imshow('YOLOv8 Live Detector', resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


if __name__ == '__main__':
    main()