import torch
from PIL import Image
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize

from model import VAE

WEIGHTS_PATH = "../weights/encoder_weights.pth"
TEST_OUTPUT = "../data/train/output/bfb9214781697dc10986e017ebbc9884.jpeg"

# Создание модели и загрузка сохраненных весов
model = VAE(32)
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
output_image = ToPILImage()(output_tensor[0].squeeze())

# Сохранение выходного изображения
output_image.save("output_image.png")


