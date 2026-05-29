# 复现 ResNet 论文中的 degradation problem。
1. 通过对比 Plain18 与 Plain34，验证普通卷积网络在层数加深后会出现训练误差反而升高的问题；
2. 再通过对比 ResNet18 与 ResNet34，验证残差连接能够缓解深层网络优化困难，使更深网络依然能够获得更好的训练效果与测试效果。
