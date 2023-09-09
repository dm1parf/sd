# логирование
from enum import Enum
import torch

DEBUG = False
USE_VIDEO = True
SHOW_VIDEO = True

# main файл, пути к данным
DIR_PATH_INPUT = "data/input"
DIR_PATH_OUTPUT = "data/output"
DIR_NAME = "input"
TEST_PATH = "test"
INPUT_DATA_PATH_FROM_UTILS = "data/test"
DATA_LOGS = "data/logs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# main параметры
is_quantize = True
is_save = True
save_rescaled_out = True

# client

# server

# размер изображения для НС
SIZE = (512, 512)

# размер изображения для НС
SCALED_SIZE_DEFAULT = (1200, 1200)

# путь к весам модели части prediction
PREDICTION_MODEL_PATH = "prediction/model/pretrained_models/dmvfn_kitti.pkl"

# имя реальных кадров в паттерне
REAL_NAME = "real"

# количество реальных кадров, передаваемых в пайплайн
REAL = 10

# имя предсказанных кадров в паттерне
FAKE_NAME = "fake"

# количество предсказанных кадров, передаваемых в пайплайн
FAKE = 3


class Models(Enum):
    UNET = 'unet'
    SD = 'sd'
    ENCODER = 'encoder'


