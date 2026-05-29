import torch
import torch.nn as nn


class ResBlock(nn.Module): #残差块定义
    def __init__(self,in_channels,out_channels,stride = 1):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,
                            kernel_size=3,stride=stride,padding=1,bias=False)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2=nn.Conv2d(out_channels,out_channels,
                            kernel_size=3,stride=1,padding=1,bias=False)#第2层不降采样 stride必须为1 
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace = True)
        
        if stride !=1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Sequential()
            
    def forward(self,x):
        identity = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        return self.relu(x)
        
class ResNet(nn.Module): #残差网络结构定义
    def __init__(self,block,num_blocks,class_num = 10):
        super().__init__()
        self.relu = nn.ReLU(inplace = True)
        self.stem=nn.Sequential(nn.Conv2d(3,64,kernel_size=3,stride = 1,padding = 1,bias=False),
                                nn.BatchNorm2d(64),
                                self.relu
                               )
        
        self.in_channels = 64

        self.layer1 = self.makelayer(block,num_blocks[0],64,1)
        self.layer2 = self.makelayer(block,num_blocks[1],128,2)
        self.layer3 = self.makelayer(block,num_blocks[2],256,2)
        self.layer4 = self.makelayer(block,num_blocks[3],512,2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512,class_num)
        
    def makelayer(self,block,num_blocks,out_channles,stride):
        layers = []
            
        layers.append(block(self.in_channels,out_channles,stride))#第一块降采样
        self.in_channels = out_channles
        
        for _ in range(1,num_blocks):
            layers.append(block(self.in_channels,out_channles,stride=1))
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out,1)
        out = self.fc(out)
        return out
def ResNet18():#系列ResNet实现
    return ResNet(ResBlock,[2,2,2,2])
def ResNet34():
    return ResNet(ResBlock,[3,4,6,3])