import torch
import torch.nn.functional as F
from model import VAE
from torchvision.transforms import Compose, ToTensor, Resize
from test_dataset import MyDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
DIR_PATH_INPUT = "../data/train/input"
DIR_PATH_OUTPUT = "../data/train/output"

WEIGHTS_PATH = "../weights/encoder_weights.pth"

# Создание наборов данных и загрузчиков
train_transforms = Compose([
    Resize((512, 512)),
    ToTensor()
])

train_dataset = MyDataset(DIR_PATH_INPUT, DIR_PATH_OUTPUT, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

net = VAE(32).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    for idx, data in enumerate(train_loader, 0):
        imgs, _ = data
        imgs = imgs.to(device)

        # Feeding a batch of images into the network to obtain the output image, mu, and logVar
        out, mu, logVar = net(imgs)

        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
        loss = F.mse_loss(out, imgs, size_average=False)

        # Backpropagation based on the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch {}: Loss {}'.format(epoch, loss))

torch.save(net.state_dict(), WEIGHTS_PATH)
