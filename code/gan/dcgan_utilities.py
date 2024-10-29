# Based on: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

# Imports
from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data



# Function: Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



# DCGAN: Generator Code
class DCGenerator(nn.Module):
    def __init__(self, nz=64, ngf=64, nc=1):
        super(DCGenerator, self).__init__()

        # Notes
        # nz - size of latent space
        # ngf - size of feature maps in generator
        # nc - number of channels of the output

        self.generator = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )


        return


    def forward(self, inputs):

        output = self.generator(inputs)

        return output



# DCGAN: Discriminator Code
class DCDiscriminator(nn.Module):
    def __init__(self, ndf=64, nc=1, img_size=64):
        super(DCDiscriminator, self).__init__()

        # Notes
        # ndf - number of features of the discriminator
        # nc - number of input channels

        # Define init variables
        self.discriminator = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            # nn.LeakyReLU(0.2, inplace=True),
            
            
            # state size. (ndf*2) x 16 x 16
            # nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 4),
            # nn.LeakyReLU(0.2, inplace=True),
            
            
            # state size. (ndf*4) x 8 x 8
            # nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            
            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )


        # TODO: Change this hard code aux features
        aux_tensor = torch.rand(1, 1, img_size, img_size)
        aux_tensor = self.discriminator(aux_tensor)
        _in_features = aux_tensor.size(0) * aux_tensor.size(1) * aux_tensor.size(2) * aux_tensor.size(3)

        self.classifier = nn.Linear(in_features=_in_features, out_features=1)


        return


    def forward(self, inputs):
        
        output = self.discriminator(inputs)

        output = torch.reshape(output, (output.size(0), -1))

        output = self.classifier(output)

        return output



# Tests
# gen = DCGenerator()
# disc = DCDiscriminator()
# with torch.no_grad():
    # aux_tensor = torch.randn(1, 64, 1, 1) 
    # out = gen(aux_tensor)
    # out = disc(out)
# print(out.shape)