import torch

DIR_PATH_INPUT = "../data/train/input"
DIR_PATH_OUTPUT = "../data/train/output"

WEIGHTS_PATH = "weights/encoder_weights.pth"

# Определяем гиперпараметры
LATENT_DIM = 20
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')