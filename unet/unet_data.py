import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class MyDataset(Dataset):
    def __init__(self, input_dir, output_dir, transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])  # default transformation
        else:
            self.transform = transform
        self.inputs = os.listdir(input_dir)  # list of input file names
        self.outputs = os.listdir(output_dir)  # list of output file names

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.inputs[idx])
        output_path = os.path.join(self.output_dir, self.outputs[idx])
        input_image = Image.open(input_path).convert('RGB')
        output_image = Image.open(output_path).convert('RGB')
        input_image = self.transform(input_image)
        output_image = self.transform(output_image)
        return input_image, output_image
