# логирование
from enum import Enum

DEBUG = True

# main файл, пути к данным
DIR_PATH_INPUT = "data/input/test_1_frames2"
DIR_PATH_OUTPUT = "data/output/test_1_frames2"
DIR_NAME = "input"
TEST_PATH = "test"
INPUT_DATA_PATH_FROM_UTILS = "data/test"

# размер изображения для НС
SIZE = (512, 512)

# размер изображения для НС
SCALED_SIZE_DEFAULT = (1200, 1200)


class Models(Enum):
    UNET = 'unet'
    SD = 'sd'
