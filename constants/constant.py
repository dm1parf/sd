# логирование
from enum import Enum
import torch
import platform
DEBUG = True

# main файл, пути к данным
DIR_PATH_INPUT = "data/input"
DIR_PATH_OUTPUT = "data/output"
DIR_NAME = "input"
TEST_PATH = "test"
INPUT_DATA_PATH_FROM_UTILS = "data/test"
DATA_LOGS = "data/logs"
LOGS_PATH = "data/logs"
# DEVICE = "mps"
DEVICE = "mps" if (platform.system() == "Darwin") else "cuda"
DATA_PATH = "data"
INPUT_PATH = "input"
OUTPUT_PATH = "output"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# размер изображения для НС
SIZE = (512, 512)

# размер изображения для НС
SCALED_SIZE_DEFAULT = (1200, 1200)

DENOISE_STEPS = 1
IS_QUANTIZE = True

class Models(Enum):
    UNET = 'unet'
    SD = 'sd'  # 'sd' or 'sd_inp'
    ENCODER = 'encoder'
