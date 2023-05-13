import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

from unet.params import WEIGHTS_PATH
from unet.unet import UNet


class UnetRestorer:
    def __init__(self):
        self.transforms = Compose([
            Resize((512, 512)),
            ToTensor()
        ])

        self.model = UNet(n_channels=3, n_classes=1)
        self.model.load_state_dict(torch.load(WEIGHTS_PATH))
        self.model.eval()

    def restoration(self, img):
        input_tensor = self.transforms(img).unsqueeze(0)

        # Прогнозирование выходного изображения
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        # Преобразование тензора в изображение
        output_image = ToPILImage()(output_tensor.squeeze())
        output_image = cv2.cvtColor(np.asarray(output_image), cv2.COLOR_RGB2BGR)
        return output_image
