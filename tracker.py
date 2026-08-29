import ssl
import time
import cv2
from ultralytics import YOLOWorld

# Bypass SSL verification on macOS for CLIP weights
ssl._create_default_https_context = ssl._create_unverified_context


def main():
    print("[INFO] Loading YOLO-World model...")
    model = YOLOWorld("yolov8s-worldv2.pt")

    target_prompts = ["black coffee mug", "cell phone", "eyeglasses"]
    print(f"[INFO] Setting target classes: {target_prompts}")
    model.set_classes(target_prompts)

    # Use AVFOUNDATION backend for macOS
    print("[INFO] Initializing webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    # Brief delay to allow hardware sensor warmup
    time.sleep(1.0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Starting video stream... Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        
        # Skip warm-up frame drops safely instead of breaking immediately
        if not success or frame is None:
            continue

        results = model.track(
            source=frame,
            device="mps",
            tracker="bytetrack.yaml",
            persist=True,
            conf=0.25,
            verbose=False
        )

        annotated_frame = results[0].plot()
        cv2.imshow("Open-Vocab Tracker", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()