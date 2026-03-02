"""YOLOv5 wrapper for object detection."""

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

# Add yolov5 to path
YOLOV5_ROOT = Path(__file__).resolve().parents[1] / "thirdparty" / "yolov5"
if str(YOLOV5_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV5_ROOT))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device


class YOLOv5Wrapper:
    """YOLOv5 wrapper for object detection."""

    def __init__(
        self,
        weights: str | Path = None,
        device: str = "",
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 1000,
        imgsz: int = 640,
    ):
        """Initialize YOLOv5 detector.

        Args:
            weights: Path to model weights file (.pt)
            device: Device to use (e.g., '0', 'cpu', '')
            conf_thres: Confidence threshold
            iou_thres: NMS IOU threshold
            max_det: Maximum detections per image
            imgsz: Inference size (pixels)
        """
        if weights is None:
            weights = YOLOV5_ROOT / "weights" / "yolov5m.pt"

        self.device = select_device(device)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.imgsz = imgsz

        # Load model
        self.model = DetectMultiBackend(weights, device=self.device)
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        self.imgsz = self.check_img_size(self.imgsz, s=self.stride)

        # Warmup
        self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))

    def check_img_size(self, imgsz: int, s: int = 32) -> int:
        """Verify imgsz is a multiple of stride."""
        return int(np.ceil(imgsz / s)) * s

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        """Preprocess image for inference.

        Args:
            img: Input image (BGR format from cv2)

        Returns:
            Preprocessed image and (ratio, padding)
        """
        # Letterbox resize
        im, ratio, (padw, padh) = letterbox(img, new_shape=self.imgsz, auto=False, scaleup=False)
        # HWC to CHW, BGR to RGB
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        return im, (ratio, (padw, padh))

    @torch.no_grad()
    def detect(self, images: List[np.ndarray] | np.ndarray) -> List[List[dict]]:
        """Run detection on image(s).

        Args:
            images: Single image (H,W,C) or list of images

        Returns:
            List of detections per image. Each detection is a dict with:
                - bbox: [x1, y1, x2, y2] in original image coordinates
                - conf: confidence score
                - cls: class id
                - label: class name
        """
        single = isinstance(images, np.ndarray)
        if single:
            images = [images]

        results = []
        for img in images:
            # Preprocess
            im, (ratio, pad) = self.preprocess(img)

            # To tensor
            im = torch.from_numpy(im).to(self.device)
            im = im.float() / 255.0
            if len(im.shape) == 3:
                im = im[None]

            # Inference
            pred = self.model(im, augment=False, visualize=False)

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=self.max_det)

            # Process results
            img_results = []
            for det in pred:
                if len(det):
                    # Rescale boxes from imgsz to original image
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        x1, y1, x2, y2 = [float(x) for x in xyxy]
                        img_results.append({
                            "bbox": [x1, y1, x2, y2],
                            "conf": float(conf),
                            "cls": int(cls),
                            "label": self.names[int(cls)],
                        })
            results.append(img_results)

        return results[0] if single else results


def main():
    """Test the YOLOv5 wrapper."""
    import os

    # Find or create test image
    test_image = YOLOV5_ROOT / "data" / "images" / "bus.jpg"
    if not test_image.exists():
        # Create a simple test image with some shapes
        print(f"Test image not found, creating a simple test image...")
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 0), -1)
        cv2.circle(img, (450, 200), 50, (0, 255, 0), -1)
        print(f"Created test image: {img.shape}")
    else:
        # Load image
        img = cv2.imread(str(test_image))
        if img is None:
            print(f"Failed to load image: {test_image}")
            return
        print(f"Loaded image: {img.shape}")

    # Initialize detector
    weights_path = YOLOV5_ROOT / "weights" / "yolov5m.pt"
    if not weights_path.exists():
        print(f"Weights not found: {weights_path}")
        print("Download with:")
        print(f"  curl -L -o {weights_path} https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt")
        return

    detector = YOLOv5Wrapper(weights=weights_path, device="cpu")
    print(f"Model loaded: {detector.names}")

    # Run detection
    results = detector.detect(img)

    print(f"\nDetected {len(results)} objects:")
    for i, det in enumerate(results, 1):
        print(f"  {i}. {det['label']} ({det['conf']:.2f}): {det['bbox']}")

    # Draw results
    for det in results:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{det['label']} {det['conf']:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Save result
    output_dir = Path(__file__).parents[1] / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "yolov5_result.jpg"
    cv2.imwrite(str(output_path), img)
    print(f"\nResult saved to: {output_path}")


if __name__ == "__main__":
    main()
