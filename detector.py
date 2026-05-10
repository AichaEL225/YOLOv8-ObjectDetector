import cv2
import os
from ultralytics import YOLO


# ─── Config ───────────────────────────────────────────────
IMAGE_PATH     = 'images/kitchen.jpg'
MODEL_SIZE     = 'yolov8s.pt'
CONFIDENCE     = 0.5
SAVE_RESULT    = True
RESULTS_FOLDER = 'results'
# ──────────────────────────────────────────────────────────


def load_model(model_size):
    print(f"Loading model: {model_size}")
    return YOLO(model_size)


def detect_image(model, image_path, confidence):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    print(f"Running detection on: {image_path}")
    results = model(image_path, conf=confidence)
    return results


def draw_and_show(results, original_image, save=True, save_folder='results'):
    os.makedirs(save_folder, exist_ok=True)

    for i, result in enumerate(results):
        # Print detections to console
        print(f"\n{'='*40}")
        print(f"Detected {len(result.boxes)} object(s):")
        for box in result.boxes:
            class_id   = int(box.cls)
            label      = result.names[class_id]
            confidence = float(box.conf)
            print(f"  - {label}: {round(confidence * 100, 1)}%")
        print(f"{'='*40}\n")

        # Draw boxes on annotated frame
        annotated = result.plot()

        # Resize annotated back to original image size
        orig_h, orig_w = original_image.shape[:2]
        annotated = cv2.resize(annotated, (orig_w, orig_h))

        # Save result
        if save:
            save_path = os.path.join(save_folder, f'result_{i}.jpg')
            cv2.imwrite(save_path, annotated)
            print(f"Result saved to: {save_path}")

        # Show image in original size
        cv2.namedWindow('YOLOv8 Object Detector', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('YOLOv8 Object Detector', orig_w, orig_h)
        cv2.imshow('YOLOv8 Object Detector', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    # Load original image first to get its real size
    original_image = cv2.imread(IMAGE_PATH)
    if original_image is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    print(f"Image size: {original_image.shape[1]}x{original_image.shape[0]}")

    model   = load_model(MODEL_SIZE)
    results = detect_image(model, IMAGE_PATH, CONFIDENCE)
    draw_and_show(results, original_image, save=SAVE_RESULT, save_folder=RESULTS_FOLDER)


if __name__ == '__main__':
    main()