# Imports
import torch
from torch import tensor
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torchvision import transforms
import os

datasets = ['CrossMatch', 'Digital_Persona', 'GreenBit', 'Hi_Scan', 'Time_Series']
materials = ['Body_Double', 'Ecoflex', 'Playdoh', 'Ecoflex_00_50', 'WoodGlue', 'Gelatine', 'Latex', 'real']


# Class: Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, root, dataset, material, img_size, cropped=True):
        if cropped:
            txt_file = f'{material}_c.txt'
        else:
            txt_file = f'{material}.txt'

        file_path = os.path.join(root, dataset, txt_file)

        assert os.path.exists(file_path), f'{file_path} does not exist'

        self.img_size = img_size
        self.ratio = 1.1

        with open(file_path, 'r') as file:
            self.paths = file.read().splitlines()

        self.paths.sort()

        self.transform = transforms.Compose([
            transforms.RandomCrop(img_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert('L')
        img = ImageOps.pad(img, (int(self.img_size*self.ratio), int(self.img_size*self.ratio)), color=255)

        tensor_img = self.transform(img)

        # Put images between [-1, 1]
        tensor_img = torch.clamp(((tensor_img * 2) - 1), min=-1, max=1)

        return tensor_img, img_path.split('_')[0]

