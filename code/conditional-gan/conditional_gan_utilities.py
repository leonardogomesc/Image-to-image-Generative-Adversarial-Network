# Based on: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

# Imports
from __future__ import print_function
from collections import OrderedDict

# PyTorch
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



# Class UNet (https://github.com/mateuszbuda/brain-segmentation-pytorch)
class UNet(nn.Module):

    def __init__(self, in_channels=2, out_channels=1, height=64, width=64, init_features=32):
        super(UNet, self).__init__()

        # Define init variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = height
        self.width = width

        # Build U-Net
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features * 8), features * 8, name="dec4")
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4), features * 4, name="dec3")
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2), features * 2, name="dec2")
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, input_1, input_2):

        # Merge both inputs
        x = torch.cat((input_1, input_2), dim=1)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        # dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        # dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        # dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        # dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return torch.tanh(self.conv(dec1))



    # Method: UNet basic-block
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )



# DCGAN: Discriminator Code
class DCDiscriminator(nn.Module):
    def __init__(self, ndf=64, nc=2, img_size=64):
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
        aux_tensor = torch.rand(1, nc, img_size, img_size)
        aux_tensor = self.discriminator(aux_tensor)
        _in_features = aux_tensor.size(0) * aux_tensor.size(1) * aux_tensor.size(2) * aux_tensor.size(3)

        self.classifier = nn.Linear(in_features=_in_features, out_features=1)


        return


    def forward(self, input_1, input_2):

        inputs = torch.cat((input_1, input_2), dim=1)
        
        output = self.discriminator(inputs)

        output = torch.reshape(output, (output.size(0), -1))

        output = self.classifier(output)

        return output



# Tests
# gen = UNet()
# disc = DCDiscriminator()
# with torch.no_grad():
    # aux_tensor = torch.randn(1, 1, 64, 64)
    # out = gen(aux_tensor, aux_tensor)
    # out = disc(out, out)

# print(out.shape)