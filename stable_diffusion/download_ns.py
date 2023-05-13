import torch
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from stable_diffusion.constant import HUGGINGFACE_TOKEN, PRETRAINED_MODEL_NAME_OR_PATH, TORCH_DEVICE


def load_sd():
    # Инициализация автоэнкодера с KL-дивергенцией из заранее обученной модели
    vae = AutoencoderKL.from_pretrained(
            PRETRAINED_MODEL_NAME_OR_PATH, subfolder="vae", use_auth_token=HUGGINGFACE_TOKEN
    ).to(TORCH_DEVICE)

    # Инициализация условной модели U-Net из заранее обученной модели
    unet = UNet2DConditionModel.from_pretrained(
            PRETRAINED_MODEL_NAME_OR_PATH, subfolder="unet", use_auth_token=HUGGINGFACE_TOKEN
    ).to(TORCH_DEVICE)

    # Инициализация планировщика Pseudo-нормального диффузионного процесса (Pseudo-Normal Diffusion Process Scheduler)
    scheduler = PNDMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
            num_train_timesteps=1000, skip_prk_steps=True
    )

    # Инициализация текстовой модели CLIP из заранее обученной модели
    text_encoder = CLIPTextModel.from_pretrained(
            PRETRAINED_MODEL_NAME_OR_PATH, subfolder="text_encoder", use_auth_token=HUGGINGFACE_TOKEN,
    )

    # Инициализация токенизатора CLIPTokenizer из заранее обученной модели
    tokenizer = CLIPTokenizer.from_pretrained(
            PRETRAINED_MODEL_NAME_OR_PATH,
            subfolder="tokenizer",
            use_auth_token=HUGGINGFACE_TOKEN,
            torch_dtype=torch.float16
    )

    # Создание тензора без условия
    uncond_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    # Вычисление его эмбеддинга с помощью CLIPTextModel
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids)[0].to(TORCH_DEVICE)

    return vae, unet, scheduler, text_encoder, tokenizer, uncond_input, uncond_embeddings

# Код на Python инициализирует несколько моделей глубокого обучения из заранее
# обученных моделей, используя Hugging Face Transformers.
