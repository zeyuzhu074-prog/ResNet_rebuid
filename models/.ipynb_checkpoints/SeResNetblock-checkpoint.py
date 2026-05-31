import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Squeeze: 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation: 两个 FC 层
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: 通道加权
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=False, se_reduction=16):
        super().__init__()
        self.use_se = use_se

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Sequential()

        # SE 模块开关
        if self.use_se:
            self.se = SEBlock(out_channels, reduction=se_reduction)
        else:
            self.se = None

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        # 在残差路径内部、与 shortcut 相加之前插入 SE
        if self.use_se:
            x = self.se(x)

        x += identity
        return self.relu(x)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, class_num=10, use_se=False, se_reduction=16):
        super().__init__()
        self.use_se = use_se
        self.se_reduction = se_reduction

        self.relu = nn.ReLU(inplace=True)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            self.relu
        )

        self.in_channels = 64
        self.layer1 = self.makelayer(block, num_blocks[0], 64, 1)
        self.layer2 = self.makelayer(block, num_blocks[1], 128, 2)
        self.layer3 = self.makelayer(block, num_blocks[2], 256, 2)
        self.layer4 = self.makelayer(block, num_blocks[3], 512, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, class_num)

    def makelayer(self, block, num_blocks, out_channels, stride):
        layers = []
        # 第一个 block 可能降采样
        layers.append(block(self.in_channels, out_channels, stride,
                            use_se=self.use_se, se_reduction=self.se_reduction))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=1,
                                use_se=self.use_se, se_reduction=self.se_reduction))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out




def SEResNet18(reduction=16):
    return ResNet(ResBlock, [2, 2, 2, 2], use_se=True, se_reduction=reduction)

def SEResNet34(reduction=16):
    return ResNet(ResBlock, [3, 4, 6, 3], use_se=True, se_reduction=reduction)