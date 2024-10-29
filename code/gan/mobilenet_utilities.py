# Imports
import numpy as np
import os
import _pickle as cPickle

# PyTorch Imports
import torch
import torchvision
import torchsummary



# Class MobileNetV3 Small
class MobileNetV3Small(torch.nn.Module):
    def __init__(self, height, width, classes):
        super(MobileNetV3Small, self).__init__()

        # Define init variables
        self.height = height
        self.width = width
        self.classes = classes

        # Construct MobileNetV3Small
        self.mobilenet = torchvision.models.mobilenet_v3_small(pretrained=False, progress=True)

        self.classifier = torch.nn.Linear(in_features=1000, out_features=self.classes)


        return


    def forward(self, inputs):
        
        outputs = torch.cat((inputs, inputs, inputs), dim=1)
        
        outputs = self.mobilenet(outputs)

        outputs = self.classifier(outputs)
        

        return outputs



# Tests
# mnet = MobileNetV3Small(224, 224, 1)
# aux = torch.rand(1, 1, 224, 224)
# out = mnet(aux)
# print(out.shape)