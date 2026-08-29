import ssl
import time
import cv2
import numpy as np
from ultralytics import YOLOWorld

# Bypass SSL verification on macOS for CLIP weight downloads
ssl._create_default_https_context = ssl._create_unverified_context


def main():
    print("[INFO] Loading YOLO-World model...")
    model = YOLOWorld("yolov8s-worldv2.pt")

    # Define dynamic text prompt targets
    target_prompts = ["black coffee mug", "cell phone", "eyeglasses"]
    print(f"[INFO] Setting target classes: {target_prompts}")
    model.set_classes(target_prompts)

    # DEFINE VIRTUAL DESK ZONE (4 Polygon Corners: [x, y])
    # Default box region covering the center portion of a standard 640x480 frame
    zone_points = np.array([
        [150, 150],  # Top-Left
        [490, 150],  # Top-Right
        [490, 420],  # Bottom-Right
        [150, 420]   # Bottom-Left
    ], np.int32)

    # Set to store unique object IDs that have entered the zone
    entered_object_ids = set()

    print("[INFO] Initializing webcam...")
    cap = cv2.VideoCapture(0)
    time.sleep(1.0)  # Hardware sensor warmup delay

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Starting Zone Analytics... Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame is None:
            continue

        # Run detection + ByteTrack tracking on Metal Performance Shaders (MPS)
        results = model.track(
            source=frame,
            device="mps",
            tracker="bytetrack.yaml",
            persist=True,
            conf=0.25,
            verbose=False
        )

        # DRAW SEMI-TRANSPARENT POLYGON ZONE
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone_points], color=(0, 0, 255)) # Red fill
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [zone_points], isClosed=True, color=(0, 0, 255), thickness=2)

        # Process detections if valid tracks exist
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()      # Bounding box coordinates
            track_ids = results[0].boxes.id.cpu().numpy()     # Tracking IDs
            cls_ids = results[0].boxes.cls.cpu().numpy()       # Class indices

            for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                x1, y1, x2, y2 = box
                track_id = int(track_id)
                class_name = target_prompts[int(cls_id)] if int(cls_id) < len(target_prompts) else "object"

                # Calculate bottom-center anchor point of object box
                bottom_center_x = int((x1 + x2) / 2)
                bottom_center_y = int(y2)
                anchor_point = (bottom_center_x, bottom_center_y)

                # Point-in-Polygon Test (returns >= 0 if point is inside polygon)
                is_inside = cv2.pointPolygonTest(zone_points, anchor_point, measureDist=False) >= 0

                if is_inside:
                    entered_object_ids.add(track_id)
                    box_color = (0, 255, 0)   # Green if inside zone
                else:
                    box_color = (255, 120, 0) # Blue if outside zone

                # Draw bounding box & bottom-center anchor point
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                cv2.circle(frame, anchor_point, 5, (0, 255, 255), -1)

                # Label text: e.g. "cell phone #1"
                label = f"{class_name} #{track_id}"
                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2
                )

        # DISPLAY ANALYTICS OVERLAY COUNTER
        count_text = f"Objects Entered Zone: {len(entered_object_ids)}"
        cv2.putText(frame, count_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Open-Vocab Zone Analytics - MacBook Neo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()