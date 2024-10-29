# Imports
import time
import os

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Project Imports
from conditional_gan_utilities import weights_init, UNet, DCDiscriminator
from data_utils import CustomDataset, datasets, materials



# Set random seed for reproducibility
# manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
# print("Random Seed: ", manualSeed)
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)



# Select DEVICE (GPU vs CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(f"Device: {device}")


# Directories
results_dir = "results/conditional"
if os.path.isdir(results_dir) == False:
    os.makedirs(results_dir)


def main():

    # Go through all the datasets
    for dset in datasets:
        for mtrl in materials:
            try:
                print(f"Current Dataset: {dset} | Current Material: {mtrl}")


                # TODO making the GAN conditional is easy (just need to concatenate the inputs for the generator and discriminator)
                # TODO review hyperparameters
                EPOCHS = 3000

                # BATCH_SIZE = 16
                # BATCH_SIZE = 64
                BATCH_SIZE = 128

                IMG_SIZE = 64
                # IMG_SIZE = 128
                # IMG_SIZE = 224


                # Learning Rate Generator
                # If using DCGAN-models
                # lr_g = 2e-4
                # lr_g = 1e-5
                # lr_g = 1e-3
                # lr_g = 1e-4

                # Learning Rate Discriminator
                # If using DCGAN-models
                # lr_d = 2e-4
                # lr_d = 1e-5
                # lr_d = 1e-3
                # lr_d = 1e-4

                # Latent Space Size
                # nz = 64
                # nz = 100
                # nz = 128
                # nz = 256
                # nz = 224

                batch_print_interval = 1

                data_root = 'data/LivDet2015/train/'
                dataset = dset
                material = mtrl


                # Create save directories
                save_dir = os.path.join(results_dir, dataset, material)
                if os.path.isdir(save_dir) == False:
                    os.makedirs(save_dir)



                # Build the training-pipeline

                # UNet Generator
                netG = UNet()
                
                # Initialise weights accordingly
                netG.apply(weights_init)
                # netG.load_state_dict(torch.load('netG.pth', map_location='cpu'))
                
                # Move Generator into device (CPU vs GPU)
                netG = netG.to(device)


                # DCGAN - Based Discriminator
                netD = DCDiscriminator()
                # Initialise weights accordingly
                netD.apply(weights_init)
                # netD.load_state_dict(torch.load('netD.pth', map_location='cpu'))
                
                # Move Discriminator into device (CPU vs GPU)
                netD = netD.to(device)


                # Optimi. Generator
                # optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_g)
                optimizerG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
                
                
                # Optim. Discriminator
                # optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d)
                optimizerD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))



                train_loader = DataLoader(CustomDataset(data_root, dataset, material, IMG_SIZE, cropped=True), BATCH_SIZE, shuffle=True, num_workers=4)

                loss = torch.nn.BCEWithLogitsLoss()

                netG.train()
                netD.train()

                for epoch in range(EPOCHS):
                    running_loss_G = 0
                    running_loss_D_real = 0
                    running_loss_D_fake = 0

                    start = time.time()

                    for i, data in enumerate(train_loader):
                        real_material = data[0].to(device)
                        real_bonafide = data[1].to(device)

                        noise = torch.randn(real_bonafide.size()).to(device)
                        fake = netG(noise, real_bonafide)

                        # Update Discriminator with real batch only
                        output_real = netD(real_material, real_bonafide)
                        real_label = torch.ones_like(output_real, device=device)
                        
                        errD_real = loss(output_real, real_label)            
                        optimizerD.zero_grad()
                        errD_real.backward()
                        optimizerD.step()


                        # Update Discriminator with fake batch only
                        output_fake = netD(fake.detach(), real_bonafide)
                        fake_label = torch.zeros_like(output_fake, device=device)

                        errD_fake = loss(output_fake, fake_label)
                        optimizerD.zero_grad()
                        errD_fake.backward()
                        optimizerD.step()

                        # update G
                        output_fake = netD(fake, real_bonafide)
                        real_label = torch.ones_like(output_fake, device=device)

                        errG = loss(output_fake, real_label)

                        optimizerG.zero_grad()
                        errG.backward()
                        optimizerG.step()

                        running_loss_G += errG.item()
                        running_loss_D_real += errD_real.item()
                        running_loss_D_fake += errD_fake.item()

                        if (i + 1) % batch_print_interval == 0:
                            print(f'[{epoch + 1}, {i + 1}] G_loss: {running_loss_G / batch_print_interval} D_loss_real: {running_loss_D_real / batch_print_interval} D_loss_fake: {running_loss_D_fake / batch_print_interval}')
                            running_loss_G = 0.0
                            running_loss_D_real = 0.0
                            running_loss_D_fake = 0.0

                    if epoch % 100 == 0:
                        torch.save(netG.state_dict(), os.path.join(save_dir, f'netG_{epoch}.pth'))
                        # torch.save(netD.state_dict(), os.path.join(save_dir, f'netD_{epoch}.pth'))
                        print('Saved models!')

                    print(f'Epoch Time: {time.time() - start}')


            except Exception as e:
                print(e)
                print("Moving on to the next combination!")


if __name__ == '__main__':
    main()