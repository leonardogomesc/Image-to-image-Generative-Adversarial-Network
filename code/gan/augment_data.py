# Imports
import os
from PIL import Image
import numpy as np

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# Project Imports
from dcgan_utilities import DCGenerator
from data_utils import datasets, materials

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


# Function: Generate new image and convert it to Numpy array
def generate_img(netG, netG_weights_fname, device, noise):

    # Load the weights and put model in evaluation mode
    netG.load_state_dict(torch.load(netG_weights_fname, map_location=device))
    netG = netG.to(device)
    netG.eval()

    # Turn off the gradients to increase speed
    with torch.no_grad():

        # Obtain the generated image
        fake = netG(noise)[0]

        # Post-processing to get the correct image range
        fake = torch.clamp(((fake + 1) / 2), min=0, max=1)
        fake = transforms.ToPILImage()(fake)

    return fake



# Select DEVICE (GPU vs CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Results Directory
results = "results"

# Augmented Images Directory
out = "aug"

data_dir = 'data/LivDet2015/train/'

# Some global variables we will need
# nz - size of latent space vector
nz = 64

# netG - the DCGenerator model object
netG = DCGenerator()

e = 2900

n_images_proportion = 0.1

# Loop through datasets
for dset in datasets:
    for mtrl in materials:

        weights_path = os.path.join(results, dset, mtrl)

        # if this is a directory, we can move on
        if os.path.isdir(weights_path):

            # calculate number of images to generate
            file_path = os.path.join(data_dir, dset, f'{mtrl}.txt')


            # Check if this exists
            if os.path.exists(file_path):
            
            
                with open(file_path, 'r') as file:
                    paths = file.read().splitlines()

                n_images = int(len(paths) * n_images_proportion)

                # create save directory
                save_dir = os.path.join(out, dset, mtrl)
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)

                for im in range(n_images):
                    # noise - the noise vector we give as input to the network
                    noise = torch.randn(1, nz, 1, 1).to(device)

                    # Get the weights name
                    weights_fname = os.path.join(weights_path, f"netG_{e}.pth")

                    # Obtain the fake image
                    gen_img = generate_img(
                        netG=netG,
                        netG_weights_fname=weights_fname,
                        device=device,
                        noise=noise
                    )

                    gen_img.save(os.path.join(save_dir, f'{im+1}.png'))

# Finish statement
print("Finished.")