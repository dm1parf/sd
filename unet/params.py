# Путь к сохраненным весам модели
import platform

import torch

WEIGHTS_PATH = "weights/unet_weights.pth"

# Пути к данным
TRAIN_INPUT_PATH = "data/train/input"
TRAIN_OUTPUT_PATH = "data/train/output"
TEST_OUTPUT = "data/output/test_1_frames2_512_512_0/4_0001.png"

# Гиперпараметры
gpu_name = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(gpu_name if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
