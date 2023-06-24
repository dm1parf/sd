from constants.constant import Models
from stable_diffusion.compressor import SdCompressor
from unet.unet_restorer import UnetRestorer
from encoder.test_pipline import EncoderRestorer


def new_model(name):
    model = None
    if name == Models.SD.value:
        model = SdCompressor()
    elif name == Models.UNET.value:
        model = UnetRestorer()
    elif name == Models.ENCODER.value:
        model = EncoderRestorer()

    return model

