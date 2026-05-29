# 复现 ResNet 论文中的 degradation problem。
## 实验目标
1. 通过对比 Plain18 与 Plain34，验证普通卷积网络在层数加深后会出现训练误差反而升高的问题；
2. 再通过对比 ResNet18 与 ResNet34，验证残差连接能够缓解深层网络优化困难，使更深网络依然能够获得更好的训练效果与测试效果。
## 项目概述
1. resnet、plain训练代码（只是调utils实现的函数、具体训练代码实现在utlis中）分别在resnetTrain.py plainTrain.py
2. 数据集是CIFAR-10，位于data文件夹
3. 超参数统一设置在config.py
4. utils 文件夹存放绘制图像功能代码、以及详细训练代码（epoch具体的梯度下降实现）
5. models 文件夹是对应pilan resnet网络架构实现 和 resnet块实现
