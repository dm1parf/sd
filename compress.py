import model_factory
from common.logging_sd import configure_logger
from constants.constant import Models
from stable_diffusion.compressor import SdCompressor

sd: SdCompressor = None


def createSd(platform):
    global sd
    sd = model_factory.new_model(Models.SD.value, platform)
    # unet = model_factory.new_model(Models.UNET.value)
    # encoder = model_factory.new_model(Models.ENCODER.value)
    logger = configure_logger(__name__)


def run_coder(img):
    global sd
    return sd.quantize_img(img)


def run_decoder(img):
    global sd
    rest_img = sd.uncompress(img)
    return rest_img
    # return unet.restoration(rest_img)
