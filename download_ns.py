import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from diffusers import AutoencoderKL, PNDMScheduler, StableDiffusionUpscalePipeline, UNet2DConditionModel
from realesrgan import RealESRGANer
from transformers import CLIPTextModel, CLIPTokenizer

pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1"
torch_device = "mps"
huggingface_token = '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'

vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", use_auth_token=huggingface_token
).to(torch_device)

unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", use_auth_token=huggingface_token
).to(torch_device)

scheduler = PNDMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        num_train_timesteps=1000, skip_prk_steps=True
)

text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", use_auth_token=huggingface_token,
)

tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_auth_token=huggingface_token,
        torch_dtype=torch.float16
)

uncond_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids)[0].to(torch_device)
