from constants.constant import Models
from stable_diffusion.compressor import SdCompressor

def new_model(name, platform):
    model = None
    if name == Models.SD.value:
        model = SdCompressor(platform)


    return model

