# 复现 ResNet 论文实验和基于改进 ResNet 的 CIFAR 图像分类
## 论文实验复现目标
1. 通过对比 Plain18 与 Plain34，验证普通卷积网络在层数加深后会出现训练误差反而升高的问题；
2. 通过对比 ResNet系列 与 Plain系列，验证ResNet系列在同等深度和超参数的情况下，有更好的更好的训练效果与测试效果。
3. 通过对比 ResNet18 与 ResNet34，验证残更深网络验证残差连接存在时，更深网络不会因为深度增加而退化。
## 项目概述
1. resnet、plain训练代码（只是调utils实现的函数、具体训练代码实现在utlis中）分别在resnetTrain.py plainTrain.py
2. 数据集是CIFAR-10，位于data文件夹
3. 超参数统一设置在config.py
4. utils 文件夹存放绘制图像功能代码、以及详细训练代码（epoch具体的梯度下降实现）
5. models 文件夹是对应pilan resnet网络架构实现 和 resnet块实现
## 项目结构

```text
ResNet_rebuid/
├── data/                   # CIFAR-10 数据集（运行时自动下载）
├── models/                 # 网络结构定义
│   ├── resnet.py           # ResNet-18/34 实现
│   └── plainnet.py         # Plain-18/34 对照网络
├── utils/                  # 工具函数
│   ├── trainer.py          # 训练/验证逻辑
│   └── plot.py             # 可视化辅助
├── outputs/                # 训练日志与模型保存
├── config.py               # 超参数配置
├── resnetTrain.py          # ResNet 训练入口
├── plainTrain.py           # PlainNet 训练入口
├── train.ipynb             # 交互式训练 notebook
└── README.md               # 本文件
```
## 论文实验复现效果概览
![plian曲线](outputs/figures/论文实验复现/plainnet_compare.png)
       Plain-18 与 Plain-34 在 CIFAR-10 上的tain error 和 test error 
![plian曲线](outputs/figures/论文实验复现/resnet_compare.png)
       ResNet-18 与Resnet-34 在 CIFAR-10 上的tain error 和 test error 
![plian曲线](outputs/figures/resnet_compare_epoch.png)
       ResNet-18 与Resnet-34 在 CIFAR-10 上的tain error 和 test error
## 改进ResNet效果概览
![plian曲线](outputs/figures/优化实现/SEResNetDrop18_scheduler_epoch100_label_smoothing=0.1.png)
BestAcc: 94.88% Loss: 0.5047 Train Error: 0.01% Test Error: 5.12%

## 详细细节如下链接
https://xattp9t9q82.feishu.cn/wiki/FYr9w1rupiflqHk1XWlcXlyEnbJ?from=from_copylink
