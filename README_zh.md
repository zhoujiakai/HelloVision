# HelloVision

一个使用 Python 的计算机视觉模型集成项目。

## 功能特性

- **目标检测** - YOLOv5 封装，用于实时目标检测
- **姿态估计** - OpenPose18k 封装，用于 18 关键点人体姿态估计
- **目标跟踪** - DeepSort 封装，用于多目标跟踪

## 安装

### 环境要求

- Python 3.8+
- Conda（推荐）

### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/zhoujiakai/HelloVision.git
cd HelloVision
```

2. 创建并激活 conda 环境：
```bash
conda create -n HelloVision python=3.12
conda activate HelloVision
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 下载模型权重：

```bash
# YOLOv5 权重
curl -L -o thirdparty/yolov5/weights/yolov5m.pt https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt

# OpenPose18k 权重
curl -L https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth -o thirdparty/openpose18k/data/checkpoint_iter_370000.pth

# DeepSort 权重
curl -L -o thirdparty/deepsort/deep_sort/deep/checkpoint/resnet18-5c106cde.pth https://download.pytorch.org/models/resnet18-5c106cde.pth
```

## 项目结构

```
HelloVision/
├── src/
│   ├── configs/          # 配置文件
│   ├── wrappers/         # 模型封装类
│   │   ├── yolov5_wrapper.py
│   │   ├── openpose_wrapper.py
│   │   └── deepsort_wrapper.py
│   ├── apps/             # 演示应用
│   └── utils/            # 工具函数
├── tests/                # 测试文件
├── thirdparty/           # 第三方模型源代码
│   ├── yolov5/
│   ├── openpose18k/
│   └── deepsort/
└── requirements.txt
```

## 使用方法

每个封装类都有一个 `main()` 函数演示使用方法：

```bash
# YOLOv5 目标检测
python src/wrappers/yolov5_wrapper.py

# OpenPose 姿态估计
python src/wrappers/openpose_wrapper.py

# DeepSort 目标跟踪
python src/wrappers/deepsort_wrapper.py
```

运行演示应用：
```bash
# 视频跟踪演示
python src/apps/demo_tracking.py

# 逐帧跟踪演示
python src/apps/demo_tracking_frames.py
```

## 模型来源

- [YOLOv5](https://github.com/ultralytics/yolov5) - 目标检测
- [OpenPose18k](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) - 姿态估计
- [DeepSort](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch) - 目标跟踪

## 许可证

MIT License
