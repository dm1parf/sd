import torch
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from common.logging_sd import configure_logger
from constants.constant import Platform
from stable_diffusion.constant import HUGGINGFACE_TOKEN, PRETRAINED_MODEL_NAME_OR_PATH, TORCH_DEVICE

logger = configure_logger(__name__)


def load_sd(platform):
    if platform == Platform.SERVER:
        # Инициализация автоэнкодера с KL-дивергенцией из заранее обученной модели
        vae = getVae()
        logger.debug(f"vae initialized")

        return vae, None, None, None, None, None, None
    elif platform == Platform.CLIENT:
        vae = getVae()
        logger.debug(f"vae initialized")

        unet = getUnet()
        logger.debug(f"unet initialized")

        scheduler = getScheduler()
        logger.debug(f"scheduler initialized")

        text_encoder = getTextEncoder()
        logger.debug(f"text_encoder initialized")

        tokenizer = getTokenizer()
        logger.debug(f"tokenizer initialized")

        uncond_input, uncond_embeddings = getInputAndEmbedding(tokenizer, text_encoder)
        logger.debug(f"uncond_input & uncond_embeddings initialized")

        return vae, unet, scheduler, text_encoder, tokenizer, uncond_input, uncond_embeddings
    elif platform == Platform.MAIN:
        vae = getVae()
        logger.debug(f"vae initialized")

        unet = getUnet()
        logger.debug(f"unet initialized")

        scheduler = getScheduler()
        logger.debug(f"scheduler initialized")

        text_encoder = getTextEncoder()
        logger.debug(f"text_encoder initialized")

        tokenizer = getTokenizer()
        logger.debug(f"tokenizer initialized")

        uncond_input, uncond_embeddings = getInputAndEmbedding(tokenizer, text_encoder)
        logger.debug(f"uncond_input & uncond_embeddings initialized")

        return vae, unet, scheduler, text_encoder, tokenizer, uncond_input, uncond_embeddings
    else:
        logger.error("Wrong parameter given to 'createSd' function")
        raise ValueError


def getVae():
    # Инициализация автоэнкодера с KL-дивергенцией из заранее обученной модели
    vae = AutoencoderKL.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH, subfolder="vae", use_auth_token=HUGGINGFACE_TOKEN
    ).to(TORCH_DEVICE)
    return vae


def getUnet():
    # Инициализация условной модели U-Net из заранее обученной модели
    unet = UNet2DConditionModel.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH, subfolder="unet", use_auth_token=HUGGINGFACE_TOKEN
    ).to(TORCH_DEVICE)
    return unet


def getScheduler():
    # Инициализация планировщика Pseudo-нормального диффузионного процесса (Pseudo-Normal Diffusion Process Scheduler)
    scheduler = PNDMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        num_train_timesteps=1000, skip_prk_steps=True
    )
    return scheduler


def getTextEncoder():
    # Инициализация текстовой модели CLIP из заранее обученной модели
    text_encoder = CLIPTextModel.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH, subfolder="text_encoder", use_auth_token=HUGGINGFACE_TOKEN,
    )
    return text_encoder


def getTokenizer():
    # Инициализация токенизатора CLIPTokenizer из заранее обученной модели
    tokenizer = CLIPTokenizer.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH,
        subfolder="tokenizer",
        use_auth_token=HUGGINGFACE_TOKEN,
        torch_dtype=torch.float16
    )
    return tokenizer


def getInputAndEmbedding(tokenizer, text_encoder):
    # Создание тензора без условия
    uncond_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    # Вычисление его эмбеддинга с помощью CLIPTextModel
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids)[0].to(TORCH_DEVICE)

    return uncond_input, uncond_embeddings

# Код на Python инициализирует несколько моделей глубокого обучения из заранее
# обученных моделей, используя Hugging Face Transformers.
