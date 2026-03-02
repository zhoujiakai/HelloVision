"""OpenPose wrapper for human pose estimation."""

import math
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

# Add openpose to path
OPENPOSE_ROOT = Path(__file__).resolve().parents[1] / "thirdparty" / "openpose18k"
if str(OPENPOSE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPOSE_ROOT))

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state

# Keypoint names for the 18 keypoints
KEYPOINT_NAMES = [
    'nose', 'neck', 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
    'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank', 'r_eye',
    'l_eye', 'r_ear', 'l_ear'
]


def normalize(img, img_mean, img_scale):
    """Normalize image."""
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    """Pad image to make dimensions multiples of stride."""
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad

# Body part connections for visualization
BODY_PARTS_KPT_IDS = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9],
    [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
    [0, 15], [15, 17], [2, 16], [5, 17]
]


class OpenPoseWrapper:
    """OpenPose wrapper for human pose estimation."""

    def __init__(
        self,
        weights: str | Path = None,
        device: str = "cpu",
        base_height: int = 256,
        stride: int = 8,
    ):
        """Initialize OpenPose pose estimator.

        Args:
            weights: Path to model weights file (.pth)
            device: Device to use ('cpu' or 'cuda')
            base_height: Base height for input image
            stride: Stride for the network
        """
        if weights is None:
            weights = OPENPOSE_ROOT / "data" / "checkpoint_iter_370000.pth"

        self.device = device
        self.base_height = base_height
        self.stride = stride

        # Load model
        self.net = PoseEstimationWithMobileNet()
        checkpoint = torch.load(str(weights), map_location='cpu')
        load_state(self.net, checkpoint)

        if device == "cuda" and torch.cuda.is_available():
            self.net = self.net.cuda()
        self.net.eval()

    def infer(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on a single image.

        Args:
            img: Input image (BGR format)

        Returns:
            Tuple of (heatmaps, pafs)
        """
        img_mean = (128, 128, 128)
        img_scale = 1 / 256

        normed_img = normalize(img, img_mean, img_scale)
        height, width, _ = normed_img.shape
        scales = [1]

        avg_heatmaps = np.zeros((height, width, 19), dtype=np.float32)
        avg_pafs = np.zeros((height, width, 38), dtype=np.float32)

        for scale in scales:
            ratio = scale * self.base_height / float(height)
            scaled_img = cv2.resize(normed_img, (0, 0), fx=ratio, fy=ratio,
                                    interpolation=cv2.INTER_CUBIC)
            min_dims = [self.base_height, max(scaled_img.shape[1], self.base_height)]
            padded_img, pad = pad_width(scaled_img, self.stride, (0, 0, 0), min_dims)

            tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
            if self.device == "cuda" and torch.cuda.is_available():
                tensor_img = tensor_img.cuda()

            with torch.no_grad():
                stages_output = self.net(tensor_img)

            stage2_heatmaps = stages_output[-2]
            heatmaps = np.transpose(stage2_heatmaps.squeeze().data.cpu().numpy(), (1, 2, 0))
            heatmaps = cv2.resize(heatmaps, (0, 0), fx=self.stride, fy=self.stride,
                                  interpolation=cv2.INTER_CUBIC)
            heatmaps = heatmaps[pad[0]:heatmaps.shape[0] - pad[2],
                               pad[1]:heatmaps.shape[1] - pad[3], :]
            heatmaps = cv2.resize(heatmaps, (width, height), interpolation=cv2.INTER_CUBIC)
            avg_heatmaps = avg_heatmaps + heatmaps / len(scales)

            stage2_pafs = stages_output[-1]
            pafs = np.transpose(stage2_pafs.squeeze().data.cpu().numpy(), (1, 2, 0))
            pafs = cv2.resize(pafs, (0, 0), fx=self.stride, fy=self.stride,
                              interpolation=cv2.INTER_CUBIC)
            pafs = pafs[pad[0]:pafs.shape[0] - pad[2],
                       pad[1]:pafs.shape[1] - pad[3], :]
            pafs = cv2.resize(pafs, (width, height), interpolation=cv2.INTER_CUBIC)
            avg_pafs = avg_pafs + pafs / len(scales)

        return avg_heatmaps, avg_pafs

    def extract_poses(
        self,
        heatmaps: np.ndarray,
        pafs: np.ndarray
    ) -> List[dict]:
        """Extract poses from heatmaps and PAFs.

        Args:
            heatmaps: Heatmap output from network (H, W, 19)
            pafs: PAF output from network (H, W, 38)

        Returns:
            List of pose dictionaries, each containing:
                - keypoints: (18, 2) array of (x, y) coordinates, -1 if not detected
                - scores: (18,) array of confidence scores
                - bbox: [x1, y1, x2, y2] bounding box
                - score: overall pose score
        """
        total_keypoints_num = 0
        all_keypoints_by_type = []

        for kpt_idx in range(18):  # 19th for bg
            total_keypoints_num += extract_keypoints(
                heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num
            )

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)

        poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue

            pose_keypoints = np.ones((18, 2), dtype=np.float32) * -1
            pose_scores = np.zeros(18, dtype=np.float32)

            for kpt_id in range(18):
                if pose_entries[n][kpt_id] != -1.0:
                    pose_keypoints[kpt_id, 0] = all_keypoints[int(pose_entries[n][kpt_id]), 0]
                    pose_keypoints[kpt_id, 1] = all_keypoints[int(pose_entries[n][kpt_id]), 1]
                    pose_scores[kpt_id] = all_keypoints[int(pose_entries[n][kpt_id]), 2]

            # Compute bounding box
            valid_keypoints = pose_keypoints[pose_keypoints[:, 0] >= 0]
            if len(valid_keypoints) > 0:
                x_min = int(np.min(valid_keypoints[:, 0]))
                y_min = int(np.min(valid_keypoints[:, 1]))
                x_max = int(np.max(valid_keypoints[:, 0]))
                y_max = int(np.max(valid_keypoints[:, 1]))
                bbox = [x_min, y_min, x_max, y_max]
            else:
                bbox = [0, 0, 0, 0]

            poses.append({
                'keypoints': pose_keypoints,
                'scores': pose_scores,
                'bbox': bbox,
                'score': float(pose_entries[n][-2]),
            })

        return poses

    @torch.no_grad()
    def detect(self, images: List[np.ndarray] | np.ndarray) -> List[List[dict]]:
        """Run pose estimation on image(s).

        Args:
            images: Single image (H,W,C) or list of images

        Returns:
            List of poses per image. Each pose is a dict with:
                - keypoints: (18, 2) array of (x, y) coordinates
                - scores: (18,) array of confidence scores
                - bbox: [x1, y1, x2, y2] bounding box
                - score: overall pose score
        """
        single = isinstance(images, np.ndarray)
        if single:
            images = [images]

        results = []
        for img in images:
            heatmaps, pafs = self.infer(img)
            poses = self.extract_poses(heatmaps, pafs)
            results.append(poses)

        return results[0] if single else results


def draw_poses(img: np.ndarray, poses: List[dict], draw_bbox: bool = False) -> np.ndarray:
    """Draw poses on image.

    Args:
        img: Input image
        poses: List of pose dictionaries
        draw_bbox: Whether to draw bounding boxes

    Returns:
        Image with poses drawn
    """
    img = img.copy()

    # Colors for keypoints
    colors = [
        (255, 0, 0), (255, 0, 128), (255, 0, 255), (255, 128, 0),
        (255, 128, 128), (255, 128, 255), (128, 0, 0), (128, 0, 128),
        (128, 128, 0), (128, 128, 128), (0, 0, 0), (0, 0, 128),
        (0, 0, 255), (0, 128, 0), (0, 128, 128), (0, 255, 0),
        (0, 255, 128), (0, 255, 255)
    ]

    for pose in poses:
        keypoints = pose['keypoints']
        scores = pose['scores']

        # Draw keypoints
        for kpt_id, (x, y) in enumerate(keypoints):
            if x >= 0 and y >= 0:
                cv2.circle(img, (int(x), int(y)), 4, colors[kpt_id], -1)

        # Draw connections
        for part_ids in BODY_PARTS_KPT_IDS:
            kpt_a_idx, kpt_b_idx = part_ids
            if (keypoints[kpt_a_idx, 0] >= 0 and keypoints[kpt_a_idx, 1] >= 0 and
                    keypoints[kpt_b_idx, 0] >= 0 and keypoints[kpt_b_idx, 1] >= 0):
                pt_a = (int(keypoints[kpt_a_idx, 0]), int(keypoints[kpt_a_idx, 1]))
                pt_b = (int(keypoints[kpt_b_idx, 0]), int(keypoints[kpt_b_idx, 1]))
                cv2.line(img, pt_a, pt_b, (0, 255, 0), 2)

        # Draw bbox
        if draw_bbox and pose['bbox'] != [0, 0, 0, 0]:
            x1, y1, x2, y2 = pose['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw score
        if keypoints[0, 0] >= 0:
            cv2.putText(
                img, f"{pose['score']:.2f}",
                (int(keypoints[0, 0]) + 5, int(keypoints[0, 1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

    return img


def main():
    """Test the OpenPose wrapper."""
    import os

    # Find or create test image
    test_image = OPENPOSE_ROOT / "data" / "shake_it_off.jpg"
    if not test_image.exists():
        test_image = OPENPOSE_ROOT / "data" / "preview.jpg"

    if test_image.exists():
        img = cv2.imread(str(test_image))
        if img is None:
            print(f"Failed to load image: {test_image}")
            return
        print(f"Loaded image: {img.shape}")
    else:
        # Create a simple test image with a stick figure
        print(f"Test image not found, creating a simple test image...")
        img = np.ones((480, 640, 3), dtype=np.uint8) * 200
        # Draw a simple stick figure
        cv2.circle(img, (320, 80), 20, (100, 100, 100), -1)  # head
        cv2.line(img, (320, 100), (320, 200), (100, 100, 100), 3)  # body
        cv2.line(img, (320, 130), (280, 160), (100, 100, 100), 3)  # left arm
        cv2.line(img, (320, 130), (360, 160), (100, 100, 100), 3)  # right arm
        cv2.line(img, (320, 200), (290, 280), (100, 100, 100), 3)  # left leg
        cv2.line(img, (320, 200), (350, 280), (100, 100, 100), 3)  # right leg
        print(f"Created test image: {img.shape}")

    # Initialize pose estimator
    weights_path = OPENPOSE_ROOT / "data" / "checkpoint_iter_370000.pth"
    if not weights_path.exists():
        print(f"Weights not found: {weights_path}")
        print("Download with:")
        print(f"  curl -L https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth -o {weights_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    estimator = OpenPoseWrapper(weights=weights_path, device=device)
    print("Model loaded")

    # Run pose estimation
    poses = estimator.detect(img)

    print(f"\nDetected {len(poses)} pose(s):")
    for i, pose in enumerate(poses, 1):
        print(f"  Pose {i}: score={pose['score']:.3f}, bbox={pose['bbox']}")
        # Count valid keypoints
        valid_kpts = np.sum(pose['keypoints'][:, 0] >= 0)
        print(f"    Valid keypoints: {valid_kpts}/18")

    # Draw results
    result_img = draw_poses(img, poses, draw_bbox=True)

    # Save result
    output_dir = Path(__file__).parents[2] / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "openpose_result.jpg"
    cv2.imwrite(str(output_path), result_img)
    print(f"\nResult saved to: {output_path}")


if __name__ == "__main__":
    main()
