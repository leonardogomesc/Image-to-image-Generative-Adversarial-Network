# Imports
import time
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Project Imports
from unet_utilities import UNetGenerator, UNetDiscriminator
# from mobilenet_utilities import MobileNetV3Small
from dcgan_utilities import weights_init, DCGenerator, DCDiscriminator
from data_utils import CustomDataset, datasets, materials

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
from torchvision import transforms



# Select DEVICE (GPU vs CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# Directories
# results_dir = "results"


# Check generated images
def main():

    nz = 100

    netG = DCGenerator()
    netG.load_state_dict(torch.load('netG_2990.pth', map_location='cpu'))
    netG = netG.to(device)

    netG.eval()

    while True:

        with torch.no_grad():

            noise = torch.randn(1, nz, 1, 1).to(device)
            fake = netG(noise)[0]

            # print(fake)
            
            fake = torch.clamp(((fake + 1) / 2), min=0, max=1)
            fake = transforms.ToPILImage()(fake)
            
            fake = np.asarray(fake)
            plt.imshow(fake, cmap="gray")
            plt.show()


if __name__ == '__main__':
    main()