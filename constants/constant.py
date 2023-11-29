# логирование
import platform
from enum import Enum

import numpy as np
import torch

# from constants.constants_for_warm_up import warm_up_bytes_prediction, warm_up_bytes_sd

DEBUG = True
USE_VIDEO = True
SHOW_VIDEO = False

# main файл, пути к данным
DIR_PATH_INPUT = "data/input"
DIR_PATH_OUTPUT = "data/output"
DIR_NAME = "input"
TEST_PATH = "test"
INPUT_DATA_PATH_FROM_UTILS = "data/test"
DATA_LOGS = "data/logs"
DEVICE = "mps"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# main параметры
is_quantize = True
is_save = True
save_rescaled_out = False

# использовать ли оптимизированную SD
USE_OPTIMIZED_SD = False

# путь к изображению для оптимизации сд
IMAGE_URL = r"/workspace/sd/image_2023-11-21_13-45-10.png"

# использовать ли оптимизированный prediction
USE_OPTIMIZED_PREDICTION = False

# client
USE_PREDICTION = False

WINDOW_NAME = 'Video'

# максимальный размер очереди кадров на клиенте
QUEUE_MAXSIZE_CLIENT = 10


# server
# максимальный размер очереди кадров на сервере
QUEUE_MAXSIZE_SERVER = 100


# client-sd
# максимальный размер очереди кадров в приложении SD на клиенте
QUEUE_MAXSIZE_CLIENT_SD = 10


# client-prediction
# максимальный размер очереди кадров в приложении предсказания на клиенте
QUEUE_MAXSIZE_CLIENT_PREDICTION = 10

# максимальное количество буфера кадров для предсказаний
MAXSIZE_OF_RESTORED_IMGS_LIST = 4

# кол-во кадров, которые нужно предсказать, чтобы получить актуальный кадр
NUMBER_OF_FRAMES_TO_PREDICT = 10

# shape массива, выдаваемый sd
NDARRAY_SHAPE_AFTER_SD = (512, 512, 3)

VIDEO_CLIENT_URL = 'localhost'
VIDEO_CLIENT_PORT = 9092
SEND_VIDEO = True
SD_CLIENT_URL = 'localhost'
SD_CLIENT_PORT = 9090
PREDICTION_CLIENT_URL = 'localhost'
PREDICTION_CLIENT_PORT = 9091


# размер изображения для НС
SIZE = (512, 512)

# размер изображения для НС
SCALED_SIZE_DEFAULT = (1200, 1200)

# сжатие без потерь
# lzma, gzip, zstd или none
lossless_compression_alg = "lzma"

is_save_compress_bin = True

# путь к весам модели части prediction
PREDICTION_MODEL_PATH = "prediction/model/pretrained_models/dmvfn_city.pkl"

# путь к весам модели части оптимизированного prediction
OPTIMIZED_PREDICTION_MODEL_PATH = "prediction/model/pretrained_models/dmvfn_optimised_512_fp16"

# имя реальных кадров в паттерне
REAL_NAME = "real"

# количество реальных кадров, передаваемых в пайплайн
REAL = 10

# имя предсказанных кадров в паттерне
FAKE_NAME = "fake"

# количество предсказанных кадров, передаваемых в пайплайн
FAKE = 3


class Platform(Enum):
    MAIN = "main"
    SERVER = "server"
    CLIENT = "client"


class Models(Enum):
    UNET = 'unet'
    SD = 'sd'
    ENCODER = 'encoder'
