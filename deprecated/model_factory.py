from deprecated.constants.constant import Models
from deprecated.stable_diffusion.compressor import SdCompressor
from deprecated.unet.unet_restorer import UnetRestorer
from deprecated.encoder.test_pipline import EncoderRestorer


def new_model(name):
    model = None
    if name == Models.SD.value:
        model = SdCompressor()
    elif name == Models.UNET.value:
        model = UnetRestorer()
    elif name == Models.ENCODER.value:
        model = EncoderRestorer()

    return model

