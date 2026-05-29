import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from models.nornet import PlainNet18 #模型加载
from models.nornet import PlainNet34

from utils.trainer import train_model  #训练代码
from utils.plot import plot_experiment #绘图代码

import config


# =========================
# device
# =========================

device = config.DEVICE

print('device:', device)

transform_train = transforms.Compose([

    transforms.RandomCrop(32, padding=4), #对图像高宽进行填充 32*32->40*40

    transforms.RandomHorizontalFlip(), #对图像随机翻转

    transforms.ToTensor(),

    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),#CIFAR103通道均值
        (0.2023, 0.1994, 0.2010) #CIFAR103通道标准差
    )
])


transform_test = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])

# =========================
# dataset
# =========================

train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=False,
    transform=transform_train
)


test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=False,
    transform=transform_test
)


# =========================
# dataloader
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=2
)


test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=2
)
# =========================
# loss
# =========================

criterion = nn.CrossEntropyLoss()



history = {}

# Plain18
model = PlainNet18().to(device)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=config.LEARNING_RATE,
    momentum=config.MOMENTUM,
    weight_decay=config.WEIGHT_DECAY
)

history['Plain18'] = train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    config.EPOCHS
)

# Plain34
model = PlainNet34().to(device)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=config.LEARNING_RATE,
    momentum=config.MOMENTUM,
    weight_decay=config.WEIGHT_DECAY
)

history['Plain34'] = train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    config.EPOCHS
)

plot_experiment(
    history,
    './outputs/figures/plainnet_compare.png'
)

