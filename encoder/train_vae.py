import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from test_dataset import MyDataset
from encoder import VAE
from params import DIR_PATH_INPUT, DIR_PATH_OUTPUT, WEIGHTS_PATH, LATENT_DIM, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE

torch.cuda.empty_cache()

# Создание наборов данных и загрузчиков
train_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

train_dataset = MyDataset(DIR_PATH_INPUT, DIR_PATH_OUTPUT, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Определяем модель и функцию потерь
model = VAE(LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH))
model.eval()

# Определение функции потерь и оптимизатора
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Обучение модели
for epoch in tqdm(range(EPOCHS)):
    running_loss = 0.0
    for inputs, targets in tqdm(train_loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)

        out = outputs[0]
        loss = criterion(out, targets)
        loss.backward()

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.4f}")
    torch.save(model.state_dict(), WEIGHTS_PATH)
