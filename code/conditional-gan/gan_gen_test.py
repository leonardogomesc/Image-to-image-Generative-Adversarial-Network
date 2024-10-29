# Imports
import time
import os
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

# Project Imports
from conditional_gan_utilities import UNet

# PyTorch Imports
import torch
from torchvision import transforms



# Select DEVICE (GPU vs CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# Directories
# results_dir = "results"


# Check generated images
def main():

    netG = UNet()
    netG.load_state_dict(torch.load('netG_900.pth', map_location='cpu'))
    netG = netG.to(device)

    # netG.eval()

    real_img = Image.open("011_9_8_crop.png").convert('L')
    real_img = ImageOps.pad(real_img, (int(64 * 1.1), int(64 * 1.1)), color=255)

    # real_img = ImageOps.pad(real_img, (64, 64), color=255)

    # real_img = np.asarray(real_img)

    # plt.imshow(real_img, cmap="gray")
    # plt.show()

    transform = transforms.Compose([transforms.RandomCrop(64), transforms.ToTensor()])
    tensor_real_img = transform(real_img)

    # Put images between [-1, 1]
    tensor_real_img = torch.clamp(((tensor_real_img * 2) - 1), min=-1, max=1)

    tensor_real_img = tensor_real_img.unsqueeze(0)




    while True:

        with torch.no_grad():

            noise = torch.randn(1, 1, 64, 64).to(device)
            
            
            
            fake = netG(noise, tensor_real_img)[0]

            # print(fake)
            
            fake = torch.clamp(((fake + 1) / 2), min=0, max=1)
            fake = transforms.ToPILImage()(fake)
            
            fake = np.asarray(fake)
            plt.imshow(fake, cmap="gray")
            plt.show()


if __name__ == '__main__':
    main()