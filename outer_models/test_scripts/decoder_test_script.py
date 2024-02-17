import argparse
import configparser
import socket
import cv2
import torch
import os
import zlib
import struct
import time
import sys
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
from outer_models.util import instantiate_from_config
from outer_models.network_swinir import SwinIR
from outer_models.prediction.model.models import Model as Predictor, DMVFN, VPvI
from outer_models.realesrgan import RealESRGANer, RealESRGANModel
from basicsr.archs.rrdbnet_arch import RRDBNet


# import torch_tensorrt
"""
# pip install accelerate transformers
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from stable_diffusion.constant import HUGGINGFACE_TOKEN, PRETRAINED_MODEL_NAME_OR_PATH, TORCH_DEVICE
import inspect

unet = UNet2DConditionModel.from_pretrained(
    PRETRAINED_MODEL_NAME_OR_PATH, subfolder="unet", use_auth_token=HUGGINGFACE_TOKEN
).to(TORCH_DEVICE)

text_encoder = CLIPTextModel.from_pretrained(
    PRETRAINED_MODEL_NAME_OR_PATH, subfolder="text_encoder", use_auth_token=HUGGINGFACE_TOKEN,
)
scheduler = PNDMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
    num_train_timesteps=374,  # 1000  # 370-375
    skip_prk_steps=True
)
tokenizer = CLIPTokenizer.from_pretrained(
    PRETRAINED_MODEL_NAME_OR_PATH,
    subfolder="tokenizer",
    use_auth_token=HUGGINGFACE_TOKEN,
    torch_dtype=torch.float16
)
uncond_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
# Вычисление его эмбеддинга с помощью CLIPTextModel
with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids)[0].to(TORCH_DEVICE)

@torch.no_grad()
def denoise(latents):
    "
    Очищает зашумленные latents с помощью Unet.

    :param latents: Тензор, содержащий шумные значения latents
    :type latents: torch.Tensor

    :return: Тензор, содержащий очищенные значения latents
    :rtype: torch.Tensor
    "

    global unet
    global scheduler
    global uncond_embeddings

    latents = latents * 0.18215
    step_size = 15
    num_inference_steps = scheduler.config.get("num_train_timesteps", 1000) // step_size
    strength = 0.04
    scheduler.set_timesteps(num_inference_steps)
    offset = scheduler.config.get("steps_offset", 0)
    init_timestep = int(num_inference_steps * strength) + offset
    init_timestep = min(init_timestep, num_inference_steps)
    timesteps = scheduler.timesteps[-init_timestep]
    timesteps = torch.tensor([timesteps], dtype=torch.long, device=TORCH_DEVICE)
    extra_step_kwargs = {}
    if "eta" in set(inspect.signature(scheduler.step).parameters.keys()):
        extra_step_kwargs["eta"] = 0.9
    latents = latents.to(unet.dtype).to(TORCH_DEVICE)
    t_start = max(num_inference_steps - init_timestep + offset, 0)
    with torch.autocast('cuda'):  # cpu
        for i, t in enumerate(scheduler.timesteps[t_start:]):
            noise_pred = unet(latents, t, encoder_hidden_states=uncond_embeddings).sample
            latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
    # reset scheduler to free cached noise predictions
    scheduler.set_timesteps(1)
    return latents / 0.18215

"""

# Сигмоидальное квантование
def dequantinize_sigmoid(quant: torch.Tensor, quant_params=None):
    if not quant_params:  # Если не передавать отдельно
        quant_params = [2.41, 256.585]
    miner, scaler = quant_params

    new_img = torch.clone(quant)
    new_img = new_img.to(torch.float16)
    new_img /= scaler
    new_img = -torch.log((1 / new_img) - 1)
    new_img += miner

    return new_img


# Сжатие с DEFLATED
def deflated_decompress(latent_img: bytes):
    """Расжатие латента с помощью DEFLATE."""

    decompressor = zlib.decompressobj()
    byters = decompressor.decompress(latent_img)
    byters += decompressor.flush()

    latent_img = torch.frombuffer(byters, dtype=torch.uint8)
    latent_img = latent_img.reshape(1, 8, 32, 32)

    return latent_img


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model


def load_decoder(config, ckpt):
    """Загрузка модели VAE."""

    if os.path.isfile("vq-f16_optimized.ts"):
        traced_model = torch.jit.load("vq-f16_optimized.ts")
    else:
        config = OmegaConf.load(f"{config}")
        model = load_model_from_config(config, f"{ckpt}")
        model = model.type(torch.float16).cuda()

        import torch_tensorrt
        import pytorch_lightning as pl
        model = model.to(torch.float32)
        model.forward = model.decode
        # model = model.to_torchscript()
        model._trainer = pl.Trainer()
        inp = [torch.randn(1, 8, 32, 32, dtype=torch.float32, device='cuda')]
        traced_model = torch.jit.trace(model, inp)
        model = torch_tensorrt.compile(
            traced_model,
            inputs=[torch_tensorrt.Input((1, 8, 32, 32), dtype=torch.float32)],
            enabled_precisions={torch.float16, torch.float32},  # torch_tensorrt.dtype.half
            truncate_long_and_double=True,
        )
        # traced_model = torch.jit.trace(model, inp)
        print("Компиляция декодера завершена!")
        torch.jit.save(traced_model, "vq-f16_optimized.ts")

    return traced_model


def load_sr(pth, upscale=1, noise_aspect=0.5, height=1080, width=1920):
    """Загрузка модели SR."""

    """
    model = SwinIR(upscale=upscale, in_chans=3, img_size=128, window_size=8,
                   img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2, upsampler='', resi_connection='1conv')
    """
    """
    model = SwinIR(upscale=upscale, in_chans=3, img_size=126, window_size=7,
                   img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2, upsampler='', resi_connection='1conv')
    """
    """
    model = SwinIR(upscale=upscale, in_chans=3, img_size=64, window_size=8,
                   img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                   mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
    """
    """
    pretrained_model = torch.load(pth)
    model.load_state_dict(pretrained_model["params"])
    """

    """  # TENSORRT НЕ ИДЁТ С RRDBNet!!!
    pre_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    backend_model = torch_tensorrt.compile(
        pre_model,
        inputs=[torch_tensorrt.Input((1, 3, height//upscale, width//upscale))],
        enabled_precisions={torch_tensorrt.dtype.half}
    )
    """
    backend_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)

    dni_base = noise_aspect
    dni_weight = [dni_base, 1-dni_base]
    model = RealESRGANer(scale=upscale, model_path=pth, dni_weight=dni_weight, tile=0, tile_pad=10, pre_pad=0,
                         model=backend_model, half=False)

    return model


def decoder_pipeline(model, latent_img):
    """Пайплайн декодирования"""

    latent_img = deflated_decompress(latent_img)
    latent_img = latent_img.float().cuda()
    latent_img = dequantinize_sigmoid(latent_img)

    # (1, 8, 32, 32)
    """
    latent_img = latent_img.reshape(1, 4, 32, 64)
    for _ in range(3):
        latent_img = denoise(latent_img)
    latent_img = latent_img.reshape(1, 8, 32, 32)
    """

    # latent_img = latent_img.type(torch.float16)
    # output_img = model.decode(latent_img)
    latent_img = latent_img.float()
    output_img = model.forward(latent_img)

    output_img = output_img.to(dtype=torch.float16)

    return output_img


def sr_pipeline(model, latent_img: torch.Tensor, height: int, width: int):
    """Пайплайн сверхразрешения"""

    # new_img = torchvision.transforms.functional.resize(latent_img, [height, width],
    #                                                    interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
    # new_img = model(new_img)
    new_img = cv2.resize(latent_img, [width, height])
    new_img = model.enhance(new_img, outscale=2)[0]

    return new_img


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="outer_models/config/vq-f16.yaml",
        help="Путь конфигурации построения автокодировщика (VAE)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outer_models/ckpt/vq-f16.ckpt",
        help="Путь к весам к модели вариационного автокодировщика (VAE)",
    )
    parser.add_argument(
        "--sr_pth",
        type=str,
        # default="outer_models/pth/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth",
        # default="outer_models/pth/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth",
        # default="outer_models/pth/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth",
        default="outer_models/pth/RealESRGAN_x2plus.pth",
        help="Путь к весам к модели сверхразрешения (SR)",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=2,
        help="Показатель увеличения размера при сверхразрешении",
    )
    parser.add_argument(
        "--pred_pth",
        type=str,
        default="outer_models/prediction/pre_trained/dmvfn_city.pkl",
        help="Путь к весам к модели предсказания",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--noise_aspect",
        type=int,
        default=0.75,
        help="the seed (for reproducible sampling)",
    )

    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    opt = parser.parse_args()
    seed_everything(opt.seed)

    decoder_model = load_decoder(opt.config, opt.ckpt)

    # (1024, 2048, 3)
    pred_model = DMVFN(load_path=opt.pred_pth).cuda()  # TODO: predictor loader
    """
    pred_model.forward = pred_model.evaluate
    pred_model = torch_tensorrt.compile(
        pred_model,
        inputs=[torch_tensorrt.Input((1024, 2048, 3))],
        enabled_precisions={torch.uint8}
    )
    """
    pred_module = Predictor(pred_model)

    config_parse = configparser.ConfigParser()
    config_parse.read(os.path.join("outer_models", "test_scripts", "decoder_config.ini"))
    socket_settings = config_parse["SocketSettings"]
    screen_settings = config_parse["ScreenSettings"]
    pipeline_settings = config_parse["PipelineSettings"]
    internal_stream_settings = config_parse["InternalStreamSettings"]

    height = int(screen_settings["height"])
    width = int(screen_settings["width"])

    sr_model = load_sr(opt.sr_pth, upscale=opt.upscale, noise_aspect=opt.noise_aspect, height=height, width=width)

    enable_sr = bool(int(pipeline_settings["enable_sr"]))
    enable_predictor = bool(int(pipeline_settings["enable_predictor"]))

    internal_stream_mode = bool(int(internal_stream_settings["stream_used"]))
    if internal_stream_mode:
        internal_stream_sock_data = (internal_stream_settings["host"], int(internal_stream_settings["port"]))
        internal_socket = socket.socket()
        internal_socket.connect(internal_stream_sock_data)

    socket_host = socket_settings["address"]
    socket_port = int(socket_settings["port"])
    new_socket = socket.socket()
    new_socket.bind((socket_host, socket_port))
    new_socket.listen(1)

    cv2.destroyAllWindows()
    print("--- Ожидаем данные с кодировщика ---")
    connection, address = new_socket.accept()
    len_defer = 4

    i = 0
    predict_img = None
    while True:
        try:
            latent_image: bytes = connection.recv(len_defer)
            if not latent_image:
                connection, address = new_socket.accept()
                continue

            image_len = struct.unpack('I', latent_image)[0]
            latent_image: bytes = connection.recv(image_len)
            while len(latent_image) != image_len:
                diff = image_len - len(latent_image)
                latent_image += connection.recv(diff)
            connection.send(b'\x01')

            a_time = time.time()
            new_img: torch.tensor = decoder_pipeline(decoder_model, latent_image)
            b_time = time.time()

            if internal_stream_mode and (predict_img is not None):
                img_bytes = predict_img.tobytes()
                len_struct = struct.pack("I", len(img_bytes))
                internal_socket.send(len_struct + img_bytes)
            elif predict_img is not None:
                cv2.imshow("===", predict_img)
                cv2.waitKey(1)

            # Дальше можно в CV2.
            new_img = new_img * 255.0
            new_img = new_img.to(torch.uint8)

            new_img = new_img.squeeze(0)
            new_img = new_img.permute(1, 2, 0)
            new_img = new_img.detach()
            new_img = new_img.cpu()
            new_img = new_img.numpy()
            if new_img.any():
                new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

                c_time = time.time()
                if enable_sr:
                    new_img = sr_pipeline(sr_model, new_img, height // opt.upscale, width // opt.upscale)
                else:
                    new_img = cv2.resize(new_img, [width, height])
                d_time = time.time()

                if internal_stream_mode:
                    img_bytes = new_img.tobytes()
                    len_struct = struct.pack("I", len(img_bytes))
                    internal_socket.send(len_struct + img_bytes)
                else:
                    cv2.imshow("===", new_img)
                    cv2.waitKey(1)


                e_time = time.time()
                if enable_predictor:
                    new_img = cv2.resize(new_img, (1024 * 2, 512 * 2))
                    predict_img = pred_module.predict([new_img, new_img], 1)
                    predict_img = cv2.resize(predict_img, (width, height), interpolation=cv2.INTER_LANCZOS4)
                f_time = time.time()

                all_time = []
                decoder_pipeline_time = round(b_time - a_time, 3)
                all_time.append(decoder_pipeline_time)
                decoder_pipeline_fps = round(1 / decoder_pipeline_time, 3)
                between_decoder_sr_time = round(c_time - b_time, 3)
                all_time.append(between_decoder_sr_time)
                if between_decoder_sr_time != 0:
                    between_decoder_sr_fps = round(1 / between_decoder_sr_time, 3)
                else:
                    between_decoder_sr_fps = "oo"
                if enable_sr:
                    sr_pipeline_time = round(d_time - c_time, 3)
                    all_time.append(sr_pipeline_time)
                    sr_pipeline_fps = round(1 / sr_pipeline_time, 3)
                if enable_predictor:
                    predict_time = round(f_time - e_time, 3)
                    all_time.append(predict_time)
                    predict_fps = round(1 / predict_time, 3)
                total_time = round(sum(all_time), 3)
                total_fps = round(1 / total_time, 3)

                print(f"--- Время выполнения: {i} ---")
                print("- Декодер:", decoder_pipeline_time, "с / FPS:", decoder_pipeline_fps)
                print("- Зазор 1:", between_decoder_sr_time, "с / FPS:", between_decoder_sr_fps)
                if enable_sr:
                    print("- SR:", sr_pipeline_time, "с / FPS", sr_pipeline_fps)
                if enable_predictor:
                    print("- Предикт:", predict_time, "с / FPS", predict_fps)
                print("- Итого:", total_time, "с / FPS", total_fps)
                print()

            i += 1

            # connection.send(b'\x01')  # Если один компьютер!
        except (ConnectionResetError, socket.error):
            connection, address = new_socket.accept()
            continue


if __name__ == "__main__":
    main()
