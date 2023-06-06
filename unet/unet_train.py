import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from unet import UNet
from params import BATCH_SIZE, DEVICE, LEARNING_RATE, NUM_EPOCHS, TRAIN_INPUT_PATH, TRAIN_OUTPUT_PATH, WEIGHTS_PATH
from unet_data import MyDataset

# Создание наборов данных и загрузчиков
train_transforms = Compose([
    Resize((512, 512)),
    ToTensor()
])

train_dataset = MyDataset(TRAIN_INPUT_PATH, TRAIN_OUTPUT_PATH, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Создание модели и перенос на устройство
model = UNet(n_channels=3, n_classes=3).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH))
model.eval()

# Определение функции потерь и оптимизатора
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Обучение модели
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")
    torch.save(model.state_dict(), WEIGHTS_PATH)
