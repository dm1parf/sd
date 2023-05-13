from constants.constant import Models
from stable_diffusion.compressor import SdCompressor
from unet.unet_restorer import UnetRestorer


def new_model(name):
    model = None
    if name == Models.SD.value:
        model = SdCompressor()
    elif name == Models.UNET.value:
        model = UnetRestorer()

    return model

