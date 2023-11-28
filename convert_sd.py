from constants.constant import IMAGE_URL
from stable_diffusion.Diffusion.img2img_pipeline import Img2ImgPipeline
from stable_diffusion.Diffusion.utilities import preprocess_image
from PIL import Image

image = Image.open(IMAGE_URL)
image = image.resize((512, 512)) 
input_image = preprocess_image(image)

 # Initialize demo
demo = Img2ImgPipeline(
    scheduler="PNDM",
    denoising_steps=3,
    output_dir=r"/workspace/sd/stable_diffusion/Diffusion/output",
    version="2.1",
    hf_token=None,
    verbose=False,
    nvtx_profile=False, # разобраться, что делает
    max_batch_size=16)

# Load TensorRT engines and pytorch modules
demo.loadEngines(r"/workspace/sd/stable_diffusion/Diffusion/engine", 
                 r"/workspace/sd/stable_diffusion/Diffusion/onnx",
                 17,
                 opt_batch_size=1, 
                 opt_image_height=512, 
                 opt_image_width=512,
                 force_export=False, 
                 force_optimize=False, 
                 force_build=False,
                 static_batch=False, 
                 static_shape=True,
                 enable_refit=False, 
                 enable_all_tactics=False,
                 timing_cache=None, 
                 onnx_refit_dir=None)
demo.loadResources(512, 512, 1, None)

print("[I] Warming up ..")
for _ in range(5):
    images = demo.infer([''], [''], input_image, 512, 512, strength=0.04, warmup=True)

images = demo.infer([''], [''], input_image, 512, 512, strength=0.04, seed=None)