# 基于 ResNet34-UNet 的 CrackForest 裂缝图像分割复现

## 1. 项目简介

本项目基于经典 U-Net 语义分割网络，结合 ResNet34 预训练编码器，在 CrackForest 道路裂缝数据集上开展裂缝图像分割实验。

项目采用 ResNet34 作为编码器、U-Net 作为解码器，通过跳跃连接融合不同尺度的特征信息，实现对道路裂缝区域的像素级分割。

本项目主要完成以下工作：

- 复现 ResNet34-UNet 图像分割网络。
- 使用预训练 ResNet34 权重初始化编码器。
- 在 CrackForest 数据集上进行模型训练与验证。
- 使用 Dice、IoU 等指标评估分割效果。
- 保存最佳模型、训练日志、评价结果及预测可视化图像。

## 2. 实验环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Ubuntu 20.04 |
| Python | 3.13.11 |
| PyTorch | 2.6.0+cu124 |
| Torchvision | 0.21.0+cu124 |
| NumPy | 2.4.4 |
| OpenCV | 4.13.0 |
| GPU | NVIDIA A10 |
| CUDA | 12.4 |

## 3. 数据集介绍

本项目使用 CrackForest 道路裂缝数据集。

数据集包含道路图像及其对应的裂缝标注掩膜，用于训练模型识别图像中的裂缝区域。

数据集划分如下：

| 数据集 | 图像数量 |
|---|---:|
| 训练集 | 94 |
| 验证集 | 24 |
| 合计 | 118 |

数据集按照固定随机种子 42 划分训练集和验证集，以保证实验划分具有可复现性。

## 4. 网络模型

### 4.1 U-Net 网络结构

U-Net 是一种经典的编码器—解码器结构语义分割网络，主要由以下部分组成：

- **编码器（Encoder）**：逐步提取图像特征，降低特征图空间分辨率。
- **解码器（Decoder）**：逐步恢复特征图分辨率，生成像素级预测结果。
- **跳跃连接（Skip Connection）**：将编码器中不同层级的特征传递到解码器，保留图像细节信息。
- **输出层**：生成单通道裂缝分割结果。

### 4.2 ResNet34-UNet 改进结构

本项目采用 ResNet34 替换原始 U-Net 的普通卷积编码部分。

主要设计如下：

1. 使用 ImageNet 预训练的 ResNet34 提取图像特征。
2. 利用 U-Net 解码器逐级恢复特征图分辨率。
3. 通过跳跃连接融合编码器与解码器的多尺度特征。
4. 使用单通道输出层完成二分类裂缝分割。

模型总参数量为 **24,632,321**。

## 5. 实验配置

| 参数 | 设置 |
|---|---|
| 模型 | ResNet34-UNet |
| 最大训练轮数 | 50 |
| 实际训练轮数 | 38 |
| Batch Size | 2 |
| 优化器 | AdamW |
| 编码器学习率 | 1e-5 |
| 解码器学习率 | 1e-4 |
| Weight Decay | 1e-4 |
| 损失函数 | BCE Loss + Dice Loss |
| 损失权重 | 0.5 + 0.5 |
| 学习率调整 | ReduceLROnPlateau |
| Early Stopping | Patience = 10 |

训练过程中使用验证集监控模型性能，并保存验证效果最好的模型权重。

## 6. 实验结果

模型训练过程中，最佳验证结果出现在第 28 个 Epoch。

| 评价指标 | 最佳结果 |
|---|---:|
| Best Epoch | 28 |
| Validation Dice | 0.72018 |
| Validation IoU | 0.56272 |
| Train Loss | 0.15386 |
| Validation Loss | 0.16828 |
| Train Dice | 0.74434 |
| Train IoU | 0.59278 |

模型在验证集上取得了约 **0.7202 的 Dice 系数**和 **0.5627 的 IoU**，能够对道路图像中的裂缝区域进行有效识别。

## 7. 实验输出

实验输出包括：

- 最佳模型权重：`weights/best_model.pth`
- 最后一轮模型权重：`weights/last_model.pth`
- 训练日志：`outputs/training_log.csv`
- 训练损失曲线
- Dice 指标变化曲线
- 验证集预测掩膜
- 原图、真实标注与预测结果对比图
- 模型评价指标 CSV 文件及汇总结果

其中，预测结果和对比图用于直观分析模型对不同道路裂缝形态的分割能力。

## 8. 项目目录

```text
ResNet34-UNet/
├── dataset/                    # 数据集加载与预处理
├── models/                     # 网络模型定义
├── outputs/                    # 实验结果与可视化文件
│   └── predictions/
│       └── comparisons/        # 多样本预测对比图
├── weights/                    # 模型权重
│   ├── best_model.pth
│   └── last_model.pth
├── train.py                    # 模型训练
├── requirements.txt             # 环境依赖
└── README.md                   # 项目说明
```

注：以上为主要功能目录示意，具体文件以仓库实际内容为准。

## 9. 实验结论

本项目完成了基于 ResNet34-UNet 的道路裂缝图像分割实验。

通过引入 ResNet34 预训练编码器，模型能够利用预训练特征提取能力，并结合 U-Net 的多尺度跳跃连接结构恢复裂缝区域的空间细节。

实验结果表明，该模型在 CrackForest 验证集上取得了 0.72018 的 Dice 系数和 0.56272 的 IoU，验证了 ResNet34-UNet 在道路裂缝分割任务中的应用效果。

本项目包含模型训练、验证评估、权重保存和预测结果可视化等完整实验流程，可作为语义分割网络复现与对比实验的参考。

## 10. 参考资料

1. Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI, 2015.
2. He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. CVPR, 2016.
3. CrackForest: A Dataset for Road Crack Detection and Segmentation.
