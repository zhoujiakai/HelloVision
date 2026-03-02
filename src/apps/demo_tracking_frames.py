"""Demo: YOLOv5 + DeepSort multi-object tracking on frame sequence."""

import sys
from pathlib import Path

import cv2

# Add project root to path
SRC_ROOT = Path(__file__).resolve().parents[2]  # Go up to project root
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wrappers.yolov5_wrapper import YOLOv5Wrapper
from wrappers.deepsort_wrapper import DeepSortWrapper, draw_tracks


def main():
    # Initialize YOLOv5 detector
    yolov5_weights = SRC_ROOT / "thirdparty" / "yolov5" / "weights" / "yolov5m.pt"
    detector = YOLOv5Wrapper(weights=yolov5_weights, device="cpu", conf_thres=0.3)
    print(f"YOLOv5 loaded: {detector.names}")

    # Initialize DeepSort tracker (n_init=3 requires 3 frames to confirm track)
    deepsort_weights = SRC_ROOT / "thirdparty" / "deepsort" / "deep_sort" / "deep" / "checkpoint" / "resnet18-5c106cde.pth"
    tracker = DeepSortWrapper(model_path=deepsort_weights, device="cpu", n_init=3)
    print("DeepSort loaded (n_init=3)")

    # Load frame sequence
    frames_dir = SRC_ROOT / "thirdparty" / "yolov5" / "data" / "demo_frames"
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    print(f"\nFound {len(frame_files)} frames")

    if not frame_files:
        print(f"No frames found in {frames_dir}")
        return

    # Create output directory for results
    output_dir = SRC_ROOT / "outputs" / "tracking_frames"
    output_dir.mkdir(exist_ok=True)

    # Process each frame
    for frame_idx, frame_path in enumerate(frame_files):
        img = cv2.imread(str(frame_path))

        # Detect objects
        detections = detector.detect(img)

        # Update tracker
        tracks = tracker.update_from_yolov5(detections, img)

        # Count track states
        confirmed = sum(1 for t in tracks if t["state"] == "confirmed")
        tentative = sum(1 for t in tracks if t["state"] == "tentative")

        # Draw tracks
        result = draw_tracks(img, tracks, label_names=detector.names)

        # Add frame info
        cv2.putText(
            result,
            f"Frame {frame_idx + 1}/{len(frame_files)} | Confirmed: {confirmed} | Tentative: {tentative}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Save result
        output_path = output_dir / frame_path.name
        cv2.imwrite(str(output_path), result)

        # Print progress
        track_info = ", ".join([f"ID{t['track_id']}({t['state'][:4]})" for t in tracks])
        print(f"Frame {frame_idx + 1:2d}: {len(detections)} detections, {len(tracks)} tracks [{track_info}]")

    print(f"\nResults saved to: {output_dir}")
    print("\nTo create a video from frames, run:")
    print(f"  ffmpeg -framerate 10 -i {output_dir}/frame_%04d.jpg -c:v libx264 -pix_fmt yuv420p output.mp4")


if __name__ == "__main__":
    main()
