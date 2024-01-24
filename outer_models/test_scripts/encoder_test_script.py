import argparse
import cv2
import numpy as np
import torch
from pytorch_lightning import seed_everything
from outer_models.util import instantiate_from_config
from omegaconf import OmegaConf


# Сигмоидальное квантование
def quantinize_sigmoid(latent_img: torch.Tensor):
    miner = latent_img.min().item()

    new_img = torch.clone(latent_img)
    new_img -= miner
    new_img = 1 / (1 + torch.exp(-new_img))
    new_max = torch.max(new_img).item()
    scaler = 255 / new_max
    new_img *= scaler
    new_img = torch.round(new_img)
    new_img = new_img.to(torch.uint8)
    new_img = new_img.clamp(0, 255)

    quant_params = [miner, scaler]
    return new_img, quant_params


# Сжатие с DEFLATED
def deflated_method(latent_img: torch.tensor):
    """Сжатие латента с помощью DEFLATE."""

    numpy_img = latent_img.numpy()
    byters = numpy_img.tobytes()

    import zlib
    compresser = zlib.compressobj(level=9, method=zlib.DEFLATED)
    byter = byters
    new_min = compresser.compress(byter)
    new_min += compresser.flush()
    print("Сжатый латент:", len(new_min) / 1024, "Кб")

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


def pipeline(model, input_image):
    img = cv2.imread(input_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img = np.moveaxis(img, 2, 0)
    img = torch.from_numpy(img)

    img = img.to(torch.float32)
    img = img / 255.0
    current_shape = img.shape
    img = img.reshape(1, *current_shape)

    # Что-то с устройством можно сюда
    model = model.cpu()
    img = img.cpu()

    model.eval()
    with torch.no_grad():
        output = model.encode(img)
        latent_img, loss, info = output

        latent_img, quant_params = quantinize_sigmoid(latent_img)

        print("Размер латента:", latent_img.shape)

        latent_img = deflated_method(latent_img)

        print(type(latent_img), len(latent_img))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="outer_models/config/vq-f16.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outer_models/ckpt/vq-f16.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # Тут просто для теста, нужно заменить на нормальное получение картинки
    base = "1"
    input_image = "outer_models/img_test/{}.png".format(base)

    pipeline(model, input_image)


if __name__ == "__main__":
    main()
