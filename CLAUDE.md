# CLAUDE.md

This file serves as a central place for project-specific context and guidelines for Claude Code.

## Project Overview

<!-- Add your project description here -->

这是一个用来集成视觉模型的项目，代码使用python

## Development Guidelines

运行代码前先激活conda环境：`conda activate HelloVision`

遇到需要添加的依赖包，添加到`requirements.txt`

包装类都会有一个使用示例：`wrappers/`中的`main函数`

## Project Structure

<!-- Describe the directory structure and organization here -->

- src：项目代码
    - configs：一些配置
    - wrappers：一些包装类
    - apps：一些简单应用
    - utils：一些工具类
- tests：一些测试文件

- thirdparty：集成进来的模型源代码
    - yolov5：yolov5的模型源代码
    - openpose18k：18关节点的人体姿态估计模型源代码
    - deepsort：deepsort的模型源代码

## Important Notes

<!-- Add any important notes or context for working with this project -->

模型的原始代码仓库：

- yolov5：https://github.com/ultralytics/yolov5
- openpose18k：https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
- deepsort：https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch

自行下载权重到指定文件夹：

- yolov5权重：`curl -L -o thirdparty/yolov5/weights/yolov5m.pt https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt`

- openpose18k权重：`curl -L https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth -o thirdparty/openpose18k/data/checkpoint_iter_370000.pth`

- deepsort权重：`curl -L -o thirdparty/deepsort/deep_sort/deep/checkpoint/resnet18-5c106cde.pth https://download.pytorch.org/models/resnet18-5c106cde.pth`
