import cv2
from ultralytics import YOLO
import os
import time


# ─── Config ───────────────────────────────────────────────
MODEL_SIZE   = 'yolov8s.pt'
CONFIDENCE   = 0.5
VIDEO_PATH   = 'videos/tiger.mp4'
SAVE_OUTPUT  = True
OUTPUT_PATH  = 'results/output_video.mp4'
# ──────────────────────────────────────────────────────────


def main():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    model = YOLO(MODEL_SIZE)
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Video writer setup
    if SAVE_OUTPUT:
        os.makedirs('results', exist_ok=True)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(cap.get(cv2.CAP_PROP_FPS))
        writer = cv2.VideoWriter(OUTPUT_PATH,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (width, height))

    print("Processing video... Press Q to stop.")
    prev_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            print("Video finished.")
            break

        results   = model(frame, conf=CONFIDENCE, verbose=False)
        annotated = results[0].plot()

        # FPS display
        curr_time = time.time()
        fps_live  = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(annotated, f'FPS: {int(fps_live)}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        if SAVE_OUTPUT:
            writer.write(annotated)

        resized = cv2.resize(annotated, (1280, 720))
        cv2.imshow('YOLOv8 Video Detector', resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if SAVE_OUTPUT:
        writer.release()
        print(f"Saved to: {OUTPUT_PATH}")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()