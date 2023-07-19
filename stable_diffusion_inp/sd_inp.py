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
            
        frame = img

        gray_image = np.full((1080, 1920), 0, dtype=np.uint8)
        
        image_for_mask = frame.copy()
        
        imgray = cv2.cvtColor(image_for_mask, cv2.COLOR_BGR2GRAY)
        
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        mask_image = cv2.drawContours(gray_image, contours, -1, (255,255,255), 1)

        # mask_image = cv2.bitwise_not(mask_image) # trash


        image = cv2.resize(frame, (512,512))
        
        mask_image = cv2.resize(mask_image, (512,512))
        # cv2.imwrite('test.jpg', mask_image)
        # break

        im_pil = Image.fromarray(image)
        mask_im_pil = Image.fromarray(mask_image)

        result_tuple = self.pipe.bonch_encode(prompt=self.prompt, image=im_pil, mask_image=mask_im_pil, num_inference_steps=self.denoise_steps)
        # result_pil = self.pipe(prompt=self.prompt, image=im_pil, mask_image=mask_im_pil, num_inference_steps=self.denoise_steps).images[0]
        # cv_
        # print(result_latents[0].shape)
        # print(result_latents[1].shape)
        # image = image
        # print(result_pil.shape)
        # open_cv_image = np.array(result_pil)
        # print(open_cv_image.shape)

        # cv2.imwrite('test.jpg', open_cv_image)
        # image_latents = result_latents[0]
        # mask_latents = result_latents[1]
        # encoded_vae = result_latents[2]
        # mask = result_latents[3]
        # print(f'result_latents = {result_latents}')
        return result_tuple
    
    def uncompress(self, encoded_tuple: tuple):
        result_img = self.pipe.bonch_decode(bonch_tuple=encoded_tuple).images[0]
        return result_img