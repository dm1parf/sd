import torch
import torchvision
from omegaconf import OmegaConf
# import os
# import sys

# sys.path.append('/NIR/Docker-Volumes/Storage/Users/Parfenov/nir')

from util import load_model_from_config
# Можно и переделать под аргументы, но смысл?
# Здесь лишь показано, как создавать


base = "1"
from_img = r"outer_models\img_test\{}.png".format(base)
to_img = r"outer_models\img_test\{}_.png".format(base)

model_type = "kl-f16"  # kl-f32
config_file = r"outer_models\config\{}.yaml".format(model_type)
ckpt_file = r"outer_models\ckpt\{}.ckpt".format(model_type)

config = OmegaConf.load(config_file)
model = load_model_from_config(config, ckpt_file)

img = torchvision.io.read_image(from_img, mode=torchvision.io.ImageReadMode.RGB)
img = img.to(torch.float32)
img = img / 255.0
current_shape = img.shape
img = img.reshape(1, *current_shape)

# Тут для себя, так как какая-то несовместимость у меня с CUDA у тех моделей
# Если что, можно просто img = img.cuda(), а model в загрузке

# model = model.cpu(); img = img.cpu()
# model = model.cuda(); img = img.cuda()

# Для тестов. Размерность тензора изначальной картинки

print(img.shape)

# Без model.eval и torch.no_grad не стоит использовать модели

model.eval()
with (torch.no_grad()):
    output = model.encode(img)
    if "vq" in model_type:  # Тут не распределение, а сразу картинка
        latent_img, loss, info = output
    elif "kl" in model_type:  # Тут распределение
        latent_img = output.sample()

    # Размерность латентного пространства

    print(latent_img.shape)

    new_img = model.decode(latent_img)

    # Размерность выходной картинки

    print(new_img.shape)

    new_img = new_img * 255.0

    new_img = new_img.to(torch.uint8)
    current_shape = list(new_img.shape)[1:]
    new_img = new_img.reshape(*current_shape)
    torchvision.io.write_png(new_img, to_img, compression_level=6)

    print("END!")
