# Imports
import torch
from torch import tensor
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torchvision import transforms
import os
import random

datasets = ['CrossMatch', 'Digital_Persona', 'GreenBit', 'Hi_Scan', 'Time_Series']
materials = ['Body_Double', 'Ecoflex', 'Playdoh', 'Ecoflex_00_50', 'WoodGlue', 'Gelatine', 'Latex']
real = 'real'


# Class: Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, root, dataset, material, img_size, cropped=True):
        if cropped:
            txt_file = f'{material}_c.txt'
            real_file = f'{real}_c.txt'
        else:
            txt_file = f'{material}.txt'
            real_file = f'{real}.txt'

        file_path = os.path.join(root, dataset, txt_file)
        real_path = os.path.join(root, dataset, real_file)

        assert os.path.exists(file_path), f'{file_path} does not exist'
        assert os.path.exists(real_path), f'{real_path} does not exist'

        self.img_size = img_size
        self.ratio = 1.1

        with open(file_path, 'r') as file:
            self.paths = file.read().splitlines()

        with open(real_path, 'r') as file:
            self.real_paths = file.read().splitlines()

        self.paths.sort()
        self.real_paths.sort()


        # Get the IDs of the fake images 
        paths_ids = set()

        # Add this IDs to a set
        for fname in self.paths:
            id = self.get_id(fname)
            paths_ids.add(id)
        

        # Get the IDs of the real images
        real_ids = set()

        # Add this IDs to a set
        for fname in self.real_paths:
            id = self.get_id(fname)
            real_ids.add(id)


        # Get valid ids
        valid_ids = paths_ids.intersection(real_ids)


        # Remove non-valid ids from the paths
        # Fake
        _paths = list()
        for p in self.paths:
            if self.get_id(p) in valid_ids:
                _paths.append(p)
        
        self.paths = _paths


        # Bonafide
        _paths = list()
        for p in self.real_paths:
            if self.get_id(p) in valid_ids:
                _paths.append(p)
        
        self.real_paths = _paths

        


        # Proceed with the valid IDs
        self.real_dic = {}

        for real_p in self.real_paths:
            id = self.get_id(real_p)

            id_list = self.real_dic.get(id, [])
            id_list.append(real_p)
            self.real_dic[id] = id_list

        self.transform = transforms.Compose([
            transforms.RandomCrop(img_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def get_id(self, path):
        filename = os.path.split(path)[1]
        id = filename.split('_')[0]

        return id

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert('L')
        img = ImageOps.pad(img, (int(self.img_size*self.ratio), int(self.img_size*self.ratio)), color=255)

        tensor_img = self.transform(img)

        # Put images between [-1, 1]
        tensor_img = torch.clamp(((tensor_img * 2) - 1), min=-1, max=1)

        # real images
        id = self.get_id(img_path)
        real_img_path = random.choice(self.real_dic[id])
        real_img = Image.open(real_img_path).convert('L')
        real_img = ImageOps.pad(real_img, (int(self.img_size * self.ratio), int(self.img_size * self.ratio)), color=255)

        tensor_real_img = self.transform(real_img)

        # Put images between [-1, 1]
        tensor_real_img = torch.clamp(((tensor_real_img * 2) - 1), min=-1, max=1)

        return tensor_img, tensor_real_img