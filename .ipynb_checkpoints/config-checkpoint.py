import torch


# =========================
# device
# =========================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =========================
# training
# =========================

BATCH_SIZE = 256

EPOCHS = 30

LEARNING_RATE = 0.1

MOMENTUM = 0.9

WEIGHT_DECAY = 5e-4


# =========================
# dataset
# =========================

NUM_CLASSES = 10

IMAGE_SIZE = 32


# =========================
# paths
# =========================

CHECKPOINT_DIR = './outputs/checkpoints/'

FIGURE_DIR = './outputs/figures/'