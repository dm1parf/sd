import os
import time

import cv2
import numpy as np
from numpy import ndarray
import torch
from PIL import Image
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize

from model import VAE


WEIGHTS_PATH = "../weights/encoder_weights.pth"


# Создание модели и загрузка сохраненных весов
model = VAE()
model.load_state_dict(torch.load(WEIGHTS_PATH))
model.eval()


# Определение трансформаций
transforms = Compose([
    Resize((512, 512)),
    ToTensor()
])

# Загрузка изображения и применение трансформаций
input_image = Image.open(f"{DIR_PATH_OUTPUT}/0_image.jpg")
input_tensor = transforms(input_image).unsqueeze(0)

# Прогнозирование выходного изображения
with torch.no_grad():
    output_tensor = model(input_tensor)

# Преобразование тензора в изображение
output_image = ToPILImage()(output_tensor.squeeze())

# Сохранение выходного изображения
output_image.save("output_image.png")






