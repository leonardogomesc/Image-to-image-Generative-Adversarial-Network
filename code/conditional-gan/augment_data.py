# Imports
import os
from PIL import Image, ImageOps
import numpy as np

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# Project Imports
from conditional_gan_utilities import UNet
from data_utils import datasets, materials, real


# PyTorch Imports
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class BonafideDataset(Dataset):
    def __init__(self, root, dataset, img_size, cropped=True):
        if cropped:
            real_file = f'{real}_c.txt'
        else:
            real_file = f'{real}.txt'

        real_path = os.path.join(root, dataset, real_file)

        assert os.path.exists(real_path), f'{real_path} does not exist'

        self.img_size = img_size
        self.ratio = 1.1

        with open(real_path, 'r') as file:
            self.real_paths = file.read().splitlines()
            
        self.real_paths.sort()

        self.transform = transforms.Compose([
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.real_paths)


    def __getitem__(self, idx):
        img_path = self.real_paths[idx]
        img = Image.open(img_path).convert('L')
        img = ImageOps.pad(img, (int(self.img_size*self.ratio), int(self.img_size*self.ratio)), color=255)

        tensor_img = self.transform(img)

        # Put images between [-1, 1]
        tensor_img = torch.clamp(((tensor_img * 2) - 1), min=-1, max=1)

        return tensor_img



# Function: Generate new image and convert it to Numpy array
def generate_img(netG, netG_weights_fname, device, noise, bonafide):

    # Load the weights and put model in evaluation mode
    netG.load_state_dict(torch.load(netG_weights_fname, map_location=device))
    netG = netG.to(device)
    netG.eval()

    # Turn off the gradients to increase speed
    with torch.no_grad():

        # Obtain the generated image
        fake = netG(noise, bonafide)[0]

        # Post-processing to get the correct image range
        fake = torch.clamp(((fake + 1) / 2), min=0, max=1)
        fake = transforms.ToPILImage()(fake)

    return fake



# Select DEVICE (GPU vs CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Results Directory
results = "results/conditional"

# Augmented Images Directory
out = "aug_conditional"

data_dir = 'data/LivDet2015/train/'

IMG_SIZE = 64

# netG - the DCGenerator model object
netG = UNet()

e = 2900


# Loop through datasets
for dset in datasets:
    for mtrl in materials:
        
        # Get the weights path
        weights_path = os.path.join(results, dset, mtrl)

        # Get the weights name
        weights_fname = os.path.join(weights_path, f"netG_{e}.pth")

        # if this is a directory, we can move on
        if os.path.exists(weights_fname):

            # create save directory
            save_dir = os.path.join(out, dset, mtrl)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
                
            # dataloader with bonafide images
            bonafide_loader = DataLoader(BonafideDataset(data_dir, dset, IMG_SIZE, cropped=True), 1, shuffle=False, num_workers=4)
            

            for im, bonafide_im in enumerate(bonafide_loader):
                # noise - the noise vector we give as input to the network
                bonafide_im = bonafide_im.to(device)
                noise = torch.randn(bonafide_im.size()).to(device)

                # Obtain the fake image
                gen_img = generate_img(
                    netG=netG,
                    netG_weights_fname=weights_fname,
                    device=device,
                    noise=noise,
                    bonafide=bonafide_im
                )

                gen_img.save(os.path.join(save_dir, f'{im+1}.png'))


# Finish statement
print("Finished.")