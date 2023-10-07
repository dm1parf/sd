# логирование
from enum import Enum
import torch

DEBUG = True
USE_VIDEO = True
SHOW_VIDEO = True

# main файл, пути к данным
DIR_PATH_INPUT = "data/input"
DIR_PATH_OUTPUT = "data/output"
DIR_NAME = "input"
TEST_PATH = "test"
INPUT_DATA_PATH_FROM_UTILS = "data/test"
DATA_LOGS = "data/logs"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

# main параметры
is_quantize = True
is_save = False
save_rescaled_out = False

# client
USE_PREDICTION = True

# server

# размер изображения для НС
SIZE = (512, 512)

# размер изображения для НС
SCALED_SIZE_DEFAULT = (1200, 1200)

# путь к весам модели части prediction
PREDICTION_MODEL_PATH = "/home/danil/NIR/sd/prediction/model/pretrained_models/dmvfn_city.pkl"

# имя реальных кадров в паттерне
REAL_NAME = "real"

# количество реальных кадров, передаваемых в пайплайн
REAL = 10

# имя предсказанных кадров в паттерне
FAKE_NAME = "fake"

# количество предсказанных кадров, передаваемых в пайплайн
FAKE = 3

# максимальный размер очереди кадров на сервере
QUEUE_MAXSIZE_SERVER = 100

# максимальный размер очереди кадров на клиенте
QUEUE_MAXSIZE_CLIENT = 10


class Platform(Enum):
    MAIN = "main"
    SERVER = "server"
    CLIENT = "client"


class Models(Enum):
    UNET = 'unet'
    SD = 'sd'
    ENCODER = 'encoder'


