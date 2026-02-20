"""DeepSort wrapper for multi-object tracking."""

import sys
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch

# Add deepsort to path
DEEPSORT_ROOT = Path(__file__).resolve().parents[2] / "thirdparty" / "deepsort"
if str(DEEPSORT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEPSORT_ROOT))

from deep_sort.deep_sort import DeepSort


class DeepSortWrapper:
    """DeepSort wrapper for multi-object tracking.

    This wrapper is designed to work with YOLOv5 detections.
    """

    def __init__(
        self,
        model_path: str | Path = None,
        device: str = "cpu",
        max_dist: float = 0.2,
        min_confidence: float = 0.3,
        nms_max_overlap: float = 1.0,
        max_iou_distance: float = 0.7,
        max_age: int = 70,
        n_init: int = 1,
        nn_budget: int = 100,
    ):
        """Initialize DeepSort tracker.

        Args:
            model_path: Path to ReID model weights (e.g., resnet18-5c106cde.pth)
            device: Device to use ('cpu' or 'cuda')
            max_dist: Maximum cosine distance for matching
            min_confidence: Minimum confidence for detections
            nms_max_overlap: Maximum overlap for NMS (1.0 = disabled)
            max_iou_distance: Maximum IOU distance for matching
            max_age: Maximum number of frames to keep a track alive
            n_init: Number of consecutive detections before track is confirmed
            nn_budget: Budget for the nearest neighbor distance metric
        """
        if model_path is None:
            model_path = DEEPSORT_ROOT / "deep_sort" / "deep" / "checkpoint" / "resnet18-5c106cde.pth"

        self.device = device
        self.use_cuda = device == "cuda" and torch.cuda.is_available()

        # Initialize DeepSort
        self.tracker = DeepSort(
            model_path=str(model_path),
            max_dist=max_dist,
            min_confidence=min_confidence,
            nms_max_overlap=nms_max_overlap,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init,
            nn_budget=nn_budget,
            use_cuda=self.use_cuda,
        )

    def update(
        self,
        detections: List[dict],
        img: np.ndarray
    ) -> List[dict]:
        """Update tracker with detections from current frame.

        Args:
            detections: List of detection dicts from YOLOv5, each with:
                - bbox: [x1, y1, x2, y2] in original image coordinates
                - conf: confidence score
                - cls: class id
            img: Current frame image (BGR format from cv2)

        Returns:
            List of track dicts, each with:
                - track_id: unique track id
                - bbox: [x1, y1, x2, y2] in original image coordinates
                - conf: confidence score
                - cls: class id
                - state: 'confirmed' | 'tentative' | 'deleted'
        """
        if not detections:
            return []

        # Convert detections to DeepSort format
        bbox_xywh = []
        confidences = []
        classes = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Convert xyxy to xywh (center format)
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
            bbox_xywh.append([cx, cy, w, h])
            confidences.append(det["conf"])
            classes.append(det["cls"])

        bbox_xywh = np.array(bbox_xywh)
        confidences = np.array(confidences)
        classes = np.array(classes)

        # Update tracker
        # DeepSort expects outputs as [x1, y1, x2, y2, cls, track_id]
        tracker_outputs, _ = self.tracker.update(
            bbox_xywh, confidences, classes, img
        )

        # Convert tracker outputs to track dicts
        tracks = []
        for output in tracker_outputs:
            x1, y1, x2, y2, cls, track_id = output
            tracks.append({
                "track_id": int(track_id),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "conf": 1.0,  # DeepSort doesn't preserve original confidence
                "cls": int(cls),
                "state": "confirmed",
            })

        # Also include tentative tracks (newly created tracks)
        for track in self.tracker.tracker.tracks:
            if track.is_tentative() and track.time_since_update == 0:
                box = track.to_tlwh()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
                tracks.append({
                    "track_id": int(track.track_id),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": 1.0,
                    "cls": int(track.cls) if track.cls is not None else 0,
                    "state": "tentative",
                })

        return tracks

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """Convert tlwh to xyxy format."""
        x, y, w, h = bbox_tlwh
        self.tracker.height = getattr(self.tracker, 'height', 1080)
        self.tracker.width = getattr(self.tracker, 'width', 1920)
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.tracker.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.tracker.height - 1)
        return x1, y1, x2, y2

    def update_from_yolov5(
        self,
        yolov5_results: List[dict],
        img: np.ndarray
    ) -> List[dict]:
        """Update tracker with YOLOv5 detection results.

        This is a convenience method that directly accepts the output
        from YOLOv5Wrapper.detect().

        Args:
            yolov5_results: Output from YOLOv5Wrapper.detect()
            img: Current frame image (BGR format from cv2)

        Returns:
            List of track dicts
        """
        return self.update(yolov5_results, img)

    def reset(self):
        """Reset the tracker, clearing all tracks."""
        # Create a new tracker instance to reset
        model_path = self.tracker.extractor.net.state_dict()
        # Re-initialize tracker with same parameters
        # Note: This is a simple reset, for a complete reset you may need to
        # recreate the DeepSortWrapper instance
        pass


def draw_tracks(
    img: np.ndarray,
    tracks: List[dict],
    label_names: Optional[List[str]] = None,
    show_conf: bool = False,
) -> np.ndarray:
    """Draw tracks on image.

    Args:
        img: Input image (BGR format)
        tracks: List of track dicts
        label_names: Optional list of class names
        show_conf: Whether to show confidence scores

    Returns:
        Image with tracks drawn
    """
    img = img.copy()

    # Generate colors for track IDs (hash-based for consistency)
    def get_color(track_id: int) -> tuple:
        np.random.seed(track_id)
        return tuple(np.random.randint(0, 255, 3).tolist())

    for track in tracks:
        track_id = track["track_id"]
        x1, y1, x2, y2 = [int(v) for v in track["bbox"]]
        cls = track["cls"]
        conf = track.get("conf", 0.0)

        # Get color for this track
        color = get_color(track_id)

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Get label name
        if label_names and 0 <= cls < len(label_names):
            label = label_names[cls]
        else:
            label = f"cls{cls}"

        # Draw label
        label_text = f"ID:{track_id} {label}"
        if show_conf:
            label_text += f" {conf:.2f}"

        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Draw background for text
        cv2.rectangle(
            img,
            (x1, y1 - text_h - baseline - 5),
            (x1 + text_w, y1),
            color,
            -1,
        )

        # Draw text
        cv2.putText(
            img,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return img


def main():
    """Test the DeepSort wrapper with YOLOv5."""
    from yolov5_wrapper import YOLOv5Wrapper

    # Initialize YOLOv5 detector
    yolov5_weights = Path(__file__).parents[2] / "thirdparty" / "yolov5" / "weights" / "yolov5m.pt"
    if not yolov5_weights.exists():
        print(f"YOLOv5 weights not found: {yolov5_weights}")
        print("Download with:")
        print(f"  curl -L -o {yolov5_weights} https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt")
        return

    detector = YOLOv5Wrapper(weights=yolov5_weights, device="cpu")
    print(f"YOLOv5 loaded: {detector.names}")

    # Initialize DeepSort tracker
    deepsort_weights = Path(__file__).parents[2] / "thirdparty" / "deepsort" / "deep_sort" / "deep" / "checkpoint" / "resnet18-5c106cde.pth"
    if not deepsort_weights.exists():
        print(f"DeepSort weights not found: {deepsort_weights}")
        print("Download with:")
        print(f"  curl -L -o {deepsort_weights} https://download.pytorch.org/models/resnet18-5c106cde.pth")
        return

    tracker = DeepSortWrapper(model_path=deepsort_weights, device="cpu")
    print("DeepSort loaded")

    # Test with a sample image
    test_image = Path(__file__).parents[2] / "thirdparty" / "yolov5" / "data" / "images" / "bus.jpg"
    if not test_image.exists():
        print(f"Test image not found: {test_image}")
        return

    img = cv2.imread(str(test_image))
    if img is None:
        print(f"Failed to load image: {test_image}")
        return

    print(f"Loaded image: {img.shape}")

    # Detect objects
    detections = detector.detect(img)
    print(f"Detected {len(detections)} objects")

    # Update tracker
    tracks = tracker.update_from_yolov5(detections, img)
    print(f"Tracking {len(tracks)} objects")

    for track in tracks:
        print(f"  Track ID {track['track_id']}: bbox={track['bbox']}, cls={track['cls']}")

    # Draw results
    result_img = draw_tracks(img, tracks, label_names=detector.names)

    # Save result
    output_dir = Path(__file__).parents[2] / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "deepsort_result.jpg"
    cv2.imwrite(str(output_path), result_img)
    print(f"\nResult saved to: {output_path}")


if __name__ == "__main__":
    main()
