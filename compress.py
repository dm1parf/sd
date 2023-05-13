import model_factory
from common.logging_sd import configure_logger
from constants.constant import Models

sd = model_factory.new_model(Models.SD.value)
unet = model_factory.new_model(Models.UNET.value)

logger = configure_logger(__name__)


def run_coder(img):
    return sd.quantize_img(img)


def run_decoder(img):
    rest_img = sd.uncompress(img)
    return unet.restoration(rest_img)
