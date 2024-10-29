# Imports
import numpy as np
import _pickle as cPickle
import os
from collections import OrderedDict

# PyTorch Imports
import torch
import torch.nn as nn
import torchvision
import torchsummary



# Class UNet (https://github.com/mateuszbuda/brain-segmentation-pytorch)
class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, height, width, init_features=32):
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

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.conv(dec1))



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



# Class UNet Encoder (https://github.com/mateuszbuda/brain-segmentation-pytorch)
class UNetEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, height, width, init_features=32):
        super(UNetEncoder, self).__init__()

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

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        
        return bottleneck



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



# Class UNet (https://github.com/mateuszbuda/brain-segmentation-pytorch)
class UNetDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, height, width, init_features=32):
        super(UNetDecoder, self).__init__()

        # Define init variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = height
        self.width = width

        # Build U-Net
        features = init_features

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8), features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4), features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2), features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):

        bottleneck = self.bottleneck(x)

        dec4 = self.upconv4(bottleneck)
        
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.conv(dec1))



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



# Class UNet (https://github.com/mateuszbuda/brain-segmentation-pytorch)
class UNetGenerator(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, height=64, width=64, latent_space_sze=100, init_features=32):
        super(UNetGenerator, self).__init__()

        # Define init variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = height
        self.width = width
        self.latent_space_sze = latent_space_sze

        # Get FC-Block to convert latent space vec into a new size
        # TODO: Convert this into a more dynamic version
        self.fc_latent = nn.Linear(in_features=latent_space_sze, out_features=256*14*14)

        # Build U-Net
        features = init_features

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features * 8), features * 8, name="dec4")
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4), features * 4, name="dec3")
        
        # self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        # self.decoder2 = UNet._block((features * 2), features * 2, name="dec2")
        
        # self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        # self.decoder1 = UNet._block(features, features, name="dec1")

         # TODO: Change this hard code aux features
        self.conv = nn.Conv2d(in_channels=features * 4, out_channels=out_channels, kernel_size=1, stride=1, padding=4)


    def forward(self, x):

        x = self.fc_latent(x)

        x = torch.reshape(input=x, shape=(-1, 256, 14, 14))

        bottleneck = self.bottleneck(x)

        dec4 = self.upconv4(bottleneck)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self.decoder3(dec3)

        # dec2 = self.upconv2(dec3)
        # dec2 = self.decoder2(dec2)

        # dec1 = self.upconv1(dec2)
        # dec1 = self.decoder1(dec1)
        
        out = self.conv(dec3)
        out = torch.tanh(out)
        
        return out



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



# Class UNet Discriminator (https://github.com/mateuszbuda/brain-segmentation-pytorch)
class UNetDiscriminator(nn.Module):

    def __init__(self, in_channels, height, width, classes, init_features=32):
        super(UNetDiscriminator, self).__init__()

        # Define init variables
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.classes = classes

        # Build U-Net
        features = init_features
        
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.encoder2 = UNet._block(features, features * 2, name="enc2")
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features, features * 4, name="bottleneck")


        # Compute in_features for the classifier
        aux_tensor = torch.rand(1, self.in_channels, self.height, self.width)
        aux_tensor = self.encoder1(aux_tensor)
        # aux_tensor = self.pool1(aux_tensor)
        # aux_tensor = self.encoder2(aux_tensor)
        # aux_tensor = self.pool2(aux_tensor)
        # aux_tensor = self.encoder3(aux_tensor)
        # aux_tensor = self.pool3(aux_tensor)
        # aux_tensor = self.encoder4(aux_tensor)
        # aux_tensor = self.pool4(aux_tensor)
        aux_tensor = self.bottleneck(aux_tensor)
        _in_features = aux_tensor.size(0) * aux_tensor.size(1) * aux_tensor.size(2) * aux_tensor.size(3)

        # Create a Linear Layer to classify samples
        self.classifier = nn.Linear(in_features=_in_features, out_features=self.classes)


    def forward(self, x):
        enc1 = self.encoder1(x)
        # enc1 = self.pool1(enc1)

        # enc2 = self.encoder2(enc1)
        # enc2 = self.pool2(enc2)

        # enc3 = self.encoder3(enc2)
        # enc3 = self.pool3(enc3)

        # enc4 = self.encoder4(enc3)
        # enc4 = self.pool4(enc4)

        bottleneck = self.bottleneck(enc1)

        flattened = torch.reshape(bottleneck, (bottleneck.size(0), -1))

        output = self.classifier(flattened)

        
        return output



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



# Tests
# d_unet = UNetDiscriminator(1, 64, 64, 1)
# g_unet = UNetGenerator()
# aux = torch.rand(1, 100)
# out = g_unet(aux)
# print(out.shape)
# out = d_unet(out)
# print(out.shape)