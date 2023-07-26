import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize

from encoder.encoder import VAE
from encoder.params import WEIGHTS_PATH


class EncoderRestorer:
    def __init__(self):
        # Создание модели и загрузка сохраненных весов
        self.model = VAE(20)
        self.model.load_state_dict(torch.load(WEIGHTS_PATH))
        print(WEIGHTS_PATH)
        self.model.eval()

        # Определение трансформаций
        self.transforms = Compose([
            Resize((512, 512)),
            ToTensor()
        ])

    def restoration(self, image):
        # Загрузка изображения и применение трансформаций
        input_tensor = self.transforms(image).unsqueeze(0)

        # Прогнозирование выходного изображения
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        # Преобразование тензора в изображение
        output_image = ToPILImage()(output_tensor[0].squeeze())
        output_image = cv2.cvtColor(np.asarray(output_image), cv2.COLOR_RGB2BGR)

        # Сохранение выходного изображения
        return output_image


