from constants.constant import Models
from stable_diffusion.compressor import SdCompressor
from stable_diffusion_inp.sd_inp import SdInpCompressor
# from unet.unet_restorer import UnetRestorer
# from encoder.test_pipline import EncoderRestorer
import logging
import sys
logger = logging.getLogger('main')

def new_model(name):
    model = None
    if name == 'sd':
        model = SdCompressor()
    elif name == 'sd_inp':
        model = SdInpCompressor()
        logger.debug(f'model {model}')
    else:
        sys.exit('ERROR! wrong model type! Exiting...')
    # elif name == Models.UNET.value:
    #     model = UnetRestorer()
    # elif name == Models.ENCODER.value:
    #     model = EncoderRestorer()

    return model

