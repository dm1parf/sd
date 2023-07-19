import model_factory
from common.logging_sd import configure_logger
from constants.constant import Models


# unet = model_factory.new_model(Models.UNET.value)
# encoder = model_factory.new_model(Models.ENCODER.value)
logger = configure_logger(__name__)

model = None

def run_coder(img):
    """
    if using sd returns encoded img
    if using sd_inp returns tuple(latent_image, latent_mask)
    """
    global model # Possibly create compress class to escape using global
    if model is None:
        model = model_factory.new_model(Models.SD.value)
    return model.quantize_img(img)


def run_decoder(img, mask_latent=None, encoded_vae=None, mask=None):
    global model
    # if mask_latent is None:
    #     rest_img = model.uncompress(img)
    # else:
    #     rest_img = model.uncompress((img, mask_latent, encoded_vae, mask))
    # if type(compress_img) == tuple:
    rest_img = model.uncompress(img)
    return rest_img
    # return unet.restoration(rest_img)

# def run_decoder(img, mask_latent=None):

#     rest_img = sd.uncompress(img)
#     return rest_img
#     # return unet.restoration(rest_img)

