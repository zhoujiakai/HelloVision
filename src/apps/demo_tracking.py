"""Demo: YOLOv5 + DeepSort multi-object tracking."""

import sys
from pathlib import Path

import cv2

# Add src to path
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wrappers.yolov5_wrapper import YOLOv5Wrapper
from wrappers.deepsort_wrapper import DeepSortWrapper, draw_tracks


def main():
    # Initialize YOLOv5 detector
    yolov5_weights = SRC_ROOT.parent / "thirdparty" / "yolov5" / "weights" / "yolov5m.pt"
    detector = YOLOv5Wrapper(weights=yolov5_weights, device="cpu")
    print(f"YOLOv5 loaded: {detector.names}")

    # Initialize DeepSort tracker
    deepsort_weights = SRC_ROOT.parent / "thirdparty" / "deepsort" / "deep_sort" / "deep" / "checkpoint" / "resnet18-5c106cde.pth"
    tracker = DeepSortWrapper(model_path=deepsort_weights, device="cpu")
    print("DeepSort loaded")

    # Open video file or webcam
    video_path = SRC_ROOT.parent / "thirdparty" / "yolov5" / "data" / "images" / "bus.jpg"

    # For demo, use a single image. For video, use cv2.VideoCapture.
    img = cv2.imread(str(video_path))
    if img is None:
        print(f"Failed to load: {video_path}")
        return

    print(f"Processing frame: {img.shape}")

    # Detect objects
    detections = detector.detect(img)
    print(f"Detected {len(detections)} objects")

    # Update tracker
    tracks = tracker.update_from_yolov5(detections, img)
    print(f"Tracking {len(tracks)} objects")

    # Draw tracks
    result = draw_tracks(img, tracks, label_names=detector.names)

    # Save result
    output_dir = SRC_ROOT.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "tracking_demo.jpg"
    cv2.imwrite(str(output_path), result)
    print(f"Result saved to: {output_path}")

    # For video processing:
    # cap = cv2.VideoCapture(video_path)
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     detections = detector.detect(frame)
    #     tracks = tracker.update_from_yolov5(detections, frame)
    #     result = draw_tracks(frame, tracks, label_names=detector.names)
    #     cv2.imshow("Tracking", result)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
