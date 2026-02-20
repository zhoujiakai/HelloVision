# HelloVision

A computer vision model integration project using Python.

## Features

- **Object Detection** - YOLOv5 wrapper for real-time object detection
- **Pose Estimation** - OpenPose18k wrapper for 18-keypoint human pose estimation
- **Object Tracking** - DeepSort wrapper for multi-object tracking

## Installation

### Prerequisites

- Python 3.8+
- Conda (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/zhoujiakai/HelloVision.git
cd HelloVision
```

2. Create and activate conda environment:
```bash
conda create -n HelloVision python=3.12
conda activate HelloVision
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download model weights:

```bash
# YOLOv5 weights
curl -L -o thirdparty/yolov5/weights/yolov5m.pt https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt

# OpenPose18k weights
curl -L https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth -o thirdparty/openpose18k/data/checkpoint_iter_370000.pth

# DeepSort weights
curl -L -o thirdparty/deepsort/deep_sort/deep/checkpoint/resnet18-5c106cde.pth https://download.pytorch.org/models/resnet18-5c106cde.pth
```

## Project Structure

```
HelloVision/
├── src/
│   ├── configs/          # Configuration files
│   ├── wrappers/         # Model wrappers
│   │   ├── yolov5_wrapper.py
│   │   ├── openpose_wrapper.py
│   │   └── deepsort_wrapper.py
│   ├── apps/             # Demo applications
│   └── utils/            # Utility functions
├── tests/                # Test files
├── thirdparty/           # Third-party model source code
│   ├── yolov5/
│   ├── openpose18k/
│   └── deepsort/
└── requirements.txt
```

## Usage

Each wrapper has a `main()` function that demonstrates usage:

```bash
# YOLOv5 object detection
python src/wrappers/yolov5_wrapper.py

# OpenPose pose estimation
python src/wrappers/openpose_wrapper.py

# DeepSort object tracking
python src/wrappers/deepsort_wrapper.py
```

Run demo applications:
```bash
# Video tracking demo
python src/apps/demo_tracking.py

# Frame-by-frame tracking demo
python src/apps/demo_tracking_frames.py
```

## Model Sources

- [YOLOv5](https://github.com/ultralytics/yolov5) - Object detection
- [OpenPose18k](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) - Pose estimation
- [DeepSort](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch) - Object tracking

## License

MIT License
