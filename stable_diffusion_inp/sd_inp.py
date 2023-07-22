from diffusers import StableDiffusionInpaintPipeline
import torch
import numpy as np
import cv2
from PIL import Image

from stable_diffusion_inp.bonch_sd_pipe import BonchSDInpPipeline

class SdInpCompressor:
    
    def __init__(self):
        # self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
        self.pipe = BonchSDInpPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
        )

        self.pipe = self.pipe.to('cuda')

        self.prompt = ''

        self.denoise_steps = 5

    def quantize_img(self, img):
        
        img = prepare_img(img)

        mask_img = create_img_mask(img)
        
        result_tuple = self.pipe.bonch_encode(prompt=self.prompt, image=img, mask_image=mask_img)
        return result_tuple
    
    def uncompress(self, encoded_tuple: tuple):
        result_img = self.pipe.bonch_decode(bonch_tuple=encoded_tuple).images[0]
        return result_img

def create_img_mask(img):
    gray_image = np.full((1080, 1920), 0, dtype=np.uint8)
    # print(img)
    img = np.array(img)
    image_for_mask = img.copy()
    
    imgray = cv2.cvtColor(image_for_mask, cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask_image = cv2.drawContours(gray_image, contours, -1, (255,255,255), 1)

    mask_image = cv2.resize(mask_image, (512,512))

    mask_im_pil = Image.fromarray(mask_image)
    return mask_im_pil

def prepare_img(img):
    image = cv2.resize(img, (512,512))
    img_pil = Image.fromarray(image)
    return img_pil