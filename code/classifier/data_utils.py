# Imports
import torch
from torch import tensor
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torchvision import transforms
import os

datasets = ['CrossMatch', 'Digital_Persona', 'GreenBit', 'Hi_Scan', 'Time_Series']
materials = ['Body_Double', 'Ecoflex', 'Playdoh', 'Ecoflex_00_50', 'WoodGlue', 'Gelatine', 'Latex']


# Class: Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, root, dataset, material, img_size, resize_ratio=1.1, data_aug=True, aug_cond=True, cropped=True, test=True):

        # Read the the right .TXT with paths
        if test:
            # Check the file path of the material (PAD)
            mtrl_file_path = os.path.join(root, dataset, "Fake", material)
            
            # Check the file path of the bonafide images
            real_file_path = os.path.join(root, dataset, "Live")
        
        else:

            if cropped:
                mtrl_txt_file = f'{material}_c.txt'
                real_txt_file = 'real_c.txt'
        
            else:
                mtrl_txt_file = f'{material}.txt'
                real_txt_file = 'real.txt'


            # Check the file path of the material (PAD)
            mtrl_file_path = os.path.join(root, dataset, mtrl_txt_file)

            # Check the file path of the bonafide images
            real_file_path = os.path.join(root, dataset, real_txt_file)



        # Assert that both the material and the bonafide images exist
        mtrl_exists, real_exists = os.path.exists(mtrl_file_path), os.path.exists(real_file_path)
        # print(mtrl_exists, real_exists)
        
        assert (mtrl_exists==True and real_exists==True), f'{mtrl_file_path} or {real_file_path} do not exist'



        # Save variables that will be useful for the image operations below
        self.img_size = img_size
        self.ratio = resize_ratio


        if test:
            # Material Paths
            mtrl_file_list = [i for i in os.listdir(mtrl_file_path) if not i.startswith('.')]
            self.material_paths = [os.path.join(mtrl_file_path, i) for i in mtrl_file_list]

            # Real/Live Paths
            real_file_list = [i for i in os.listdir(real_file_path) if not i.startswith('.')]
            self.real_paths = [os.path.join(real_file_path, i) for i in real_file_list]

        else:
            # Open .TXT files
            # Material Paths
            with open(mtrl_file_path, 'r') as file:
                self.material_paths = file.read().splitlines()


            # Real Paths
            with open(real_file_path, 'r') as file:
                self.real_paths = file.read().splitlines()
        


        # Check data-augmentation parameter
        if data_aug:

            # Main base path
            if aug_cond:
                aug_root_path = os.path.join("aug_conditional", dataset)
                print("Data Augmentation with Images from cDCGAN-Generator")
            
            else:
                aug_root_path = os.path.join("aug", dataset)
                print("Data Augmentation with Images from DCGAN-Generator")
            
            # Materials
            # Create root materials path
            aug_root_materials = os.path.join(aug_root_path, material)

            # Get materials file list
            aug_materials_imgs = [i for i in os.listdir(aug_root_materials) if not i.startswith('.')]

            # Create a list to pre-append augmented materials
            aug_material_paths = [os.path.join(aug_root_materials, fname) for fname in aug_materials_imgs]

            # Append these paths to the material paths
            self.material_paths += aug_material_paths


            if not aug_cond:
                # Real
                # Create root real path
                aug_root_real = os.path.join(aug_root_path, "real")

                # Get BF images file list
                aug_real_imgs = [i for i in os.listdir(aug_root_real) if not i.startswith('.')]

                # Create a list to pre-append augmented BF
                aug_real_paths = [os.path.join(aug_root_real, fname) for fname in aug_real_imgs]

                # Append these paths to the material paths
                self.real_paths += aug_real_paths
        




        # Create all paths
        self.paths = self.material_paths + self.real_paths
        self.paths.sort()


        
        # Group of transforms to apply during training
        if test:
            self.transform = transforms.Compose([transforms.CenterCrop(img_size), transforms.ToTensor()])
        
        else:
            self.transform = transforms.Compose([transforms.RandomCrop(img_size), transforms.ToTensor()])



    def __len__(self):
        return len(self.paths)



    def __getitem__(self, idx):

        # Open image
        img_path = self.paths[idx]
        img = Image.open(img_path).convert('L')
        img = ImageOps.pad(img, (int(self.img_size*self.ratio), int(self.img_size*self.ratio)), color=255)

        tensor_img = self.transform(img)

        # Put images between [-1, 1]
        tensor_img = torch.clamp(((tensor_img * 2) - 1), min=-1, max=1)

        # Concatenate this so we have images with 3 channels
        tensor_img = torch.cat((tensor_img, tensor_img, tensor_img), dim=0)


        # Get label
        label = 0. if img_path in self.material_paths else 1.



        # Debugging purposes
        # print(img_path.split('_')[0])

        return tensor_img, label