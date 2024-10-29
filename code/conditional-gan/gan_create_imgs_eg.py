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
from data_utils import datasets, materials

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


# Set matplotlib font (default is 12)
matplotlib.rc('font', size=6)



# Function: Generate image grid
def plot_img_grid(img_list, figsize, nrows_ncols, epochs, fig_path, show_fig):

    # Create a Figure object
    fig = plt.figure(figsize=figsize) # The size of the figure is specified as (width, height) in inches
    

    # Create an ImageGrid object
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=nrows_ncols,  # creates rxc grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

    # Go through the image list
    for ax, im, epoch in zip(grid, img_list, epochs):
        # Iterating over the grid returns the Axes.
        ax.imshow(im, cmap="gray")
        ax.set_title(f"E: {epoch}")
        ax.axis('off')


    # Save figure
    plt.savefig(fig_path)


    # Show the result
    if show_fig:
        plt.show()
    

    # Clear figure
    plt.clf()

    return



# Function: Generate new image and convert it to Numpy array
def generate_img(netG, netG_weights_fname, real_img_fname, device, noise):

    # Load the weights and put model in evaluation mode
    netG.load_state_dict(torch.load(netG_weights_fname, map_location=device))
    netG = netG.to(device)
    netG.eval()


    # Load a bonafide image
    real_img = Image.open(real_img_fname).convert('L')
    real_img = ImageOps.pad(real_img, (64, 64), color=255)
    
    # Convert image to tensor
    tensor_real_img = transforms.ToTensor()(real_img)

    # Put images between [-1, 1]
    tensor_real_img = torch.clamp(((tensor_real_img * 2) - 1), min=-1, max=1)

    # Add batch dimension
    tensor_real_img = tensor_real_img.unsqueeze(0)

    # Add this to device
    tensor_real_img = tensor_real_img.to(device)

    # Turn off the gradients to increase speed
    with torch.no_grad():

        # Obtain the generated image
        fake = netG(noise, tensor_real_img)[0]

        # Post-processing to get the correct image range
        fake = torch.clamp(((fake + 1) / 2), min=0, max=1)
        fake = transforms.ToPILImage()(fake)
        
        
        # Convert it to Numpy array
        fake = np.asarray(fake)
    

    return fake



# Select DEVICE (GPU vs CPU)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")



# Epoch Range
epochs = [i for i in range(0, 2901, 200)]
# print(epochs)

# Results Directory
results = "results"

# Figures Directory
figures = "figures"


# Some global variables we will need
# netG - the DCGenerator model object
netG = UNet()

# noise - the noise vector we give as input to the network
noise = torch.randn(1, 1, 64, 64).to(device)


# Loop through datasets
for dset in datasets:

    # Pick the same real image per dataset for the conditional-GAN
    # Let's go the path of the real images in this dataset 
    real_paths = os.path.join("data/LivDet2015/train/", dset, 'real_c.txt')

    # Open file with image names
    with open(real_paths, 'r') as file:
        real_paths = file.read().splitlines()

    # The image will be the first of the list
    real_img_fname  = real_paths[0]

    for mtrl in materials:
        weights_path = os.path.join(results, "conditional", dset, mtrl)

        # if this is a directory, we can move on
        if os.path.isdir(weights_path):

            try:

                # Create an image list
                img_list = list()

                # Iterate through epochs
                for e in epochs:
                    
                    # Get the weights name
                    weights_fname = os.path.join(weights_path, f"netG_{e}.pth")
                    

                    # Obtain the fake image
                    gen_img = generate_img(
                        netG=netG,
                        netG_weights_fname=weights_fname,
                        real_img_fname=real_img_fname,
                        device=device,
                        noise=noise
                        )
                    
                    # Appen this image to the image list
                    img_list.append(gen_img)
                

                # Create figure path directory
                fig_path_dir = os.path.join(figures, "conditionalgan", dset, mtrl)
                if os.path.isdir(fig_path_dir) ==  False:
                    os.makedirs(fig_path_dir)

                # Get figure path
                fig_path = os.path.join(fig_path_dir, "figure.png")


                # Generate ImageGrid
                _ = plot_img_grid(
                    img_list=img_list,
                    figsize=(15., 2.),
                    nrows_ncols=(1, len(img_list)),
                    epochs=epochs,
                    fig_path=fig_path,
                    show_fig=False
                    )
            
            
            except:
                print(f"{weights_path} is empty.")



# Finish statement
print("Finished.")