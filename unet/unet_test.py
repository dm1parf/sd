import torch
from PIL import Image
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize

from unet import UNet
from unet.params import TEST_OUTPUT, WEIGHTS_PATH

# Создание модели и загрузка сохраненных весов
model = UNet(n_channels=3, n_classes=1)
model.load_state_dict(torch.load(WEIGHTS_PATH))
model.eval()

# Определение трансформаций
transforms = Compose([
    Resize((512, 512)),
    ToTensor()
])

# Загрузка изображения и применение трансформаций
input_image = Image.open(TEST_OUTPUT)
input_tensor = transforms(input_image).unsqueeze(0)

# Прогнозирование выходного изображения
with torch.no_grad():
    output_tensor = model(input_tensor)

# Преобразование тензора в изображение
output_image = ToPILImage()(output_tensor.squeeze())

# Сохранение выходного изображения
output_image.save("output_image.png")
