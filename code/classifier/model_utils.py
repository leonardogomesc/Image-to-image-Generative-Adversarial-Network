# Imports
import numpy as np
import _pickle as cPickle
import os


# PyTorch Imports
import torch
import torchvision
import torchsummary


# Create PyTorch Models
# Model: DenseNet 121 (Baseline)
class DenseNet121(torch.nn.Module):
    def __init__(self, channels, height, width, nr_classes):
        super(DenseNet121, self).__init__()

        # Init variables
        self.channels = channels
        self.height = height
        self.width = width
        self.nr_classes = nr_classes


        # Init modules
        # Backbone to extract features
        self.densenet121 = torchvision.models.densenet121(pretrained=True).features

        # FC-Layers
        # Compute in_features
        _in_features = torch.rand(1, self.channels, self.height, self.width)
        _in_features = self.densenet121(_in_features)
        _in_features = _in_features.size(0) * _in_features.size(1) * _in_features.size(2) * _in_features.size(3)

        # Create FC1 Layer for classification
        self.fc1 = torch.nn.Linear(in_features=_in_features, out_features=self.nr_classes)

        # Sigmoid Activation Layer
        # self.fc_sigmoid = torch.nn.Sigmoid()


        return
    

    def forward(self, inputs):
        # Compute Backbone features
        features = self.densenet121(inputs)

        # Reshape features
        features = torch.reshape(features, (features.size(0), -1))

        # FC1-Layer
        outputs = self.fc1(features)

        # Activation layer
        # outputs = self.fc_sigmoid(outputs)


        return outputs



# Model: ResNet 50 (Baseline)
class ResNet50(torch.nn.Module):
    def __init__(self, channels, height, width, nr_classes):
        super(ResNet50, self).__init__()

        # Init variables
        self.channels = channels
        self.height = height
        self.width = width
        self.nr_classes = nr_classes


        # Init modules
        # Backbone to extract features
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))

        # FC-Layers
        # Compute in_features
        _in_features = torch.rand(1, self.channels, self.height, self.width)
        _in_features = self.resnet50(_in_features)
        _in_features = _in_features.size(0) * _in_features.size(1) * _in_features.size(2) * _in_features.size(3)

        # Create FC1 Layer for classification
        self.fc1 = torch.nn.Linear(in_features=_in_features, out_features=self.nr_classes)

        # Sigmoid Activation Layer
        # self.fc_sigmoid = torch.nn.Sigmoid()



        return
    

    def forward(self, inputs):
        # Compute Backbone features
        features = self.resnet50(inputs)

        # Reshape features
        features = torch.reshape(features, (features.size(0), -1))

        # FC1-Layer
        outputs = self.fc1(features)

        # Activation layer
        # outputs = self.fc_sigmoid(outputs)


        return outputs


# Model: VGG-16 (Baseline)
class VGG16(torch.nn.Module):
    def __init__(self, channels, height, width, nr_classes):
        super(VGG16, self).__init__()

        # Init variables
        self.channels = channels
        self.height = height
        self.width = width
        self.nr_classes = nr_classes


        # Init modules
        # Backbone to extract features
        self.vgg16 = torchvision.models.vgg16(pretrained=True).features

        # FC-Layers
        # Compute in_features
        _in_features = torch.rand(1, self.channels, self.height, self.width)
        _in_features = self.vgg16(_in_features)
        _in_features = _in_features.size(0) * _in_features.size(1) * _in_features.size(2) * _in_features.size(3)

        # Create FC1 Layer for classification
        self.fc1 = torch.nn.Linear(in_features=_in_features, out_features=self.nr_classes)

        # Sigmoid Activation Layer
        # self.fc_sigmoid = torch.nn.Sigmoid()



        return
    

    def forward(self, inputs):
        # Compute Backbone features
        features = self.vgg16(inputs)

        # Reshape features
        features = torch.reshape(features, (features.size(0), -1))

        # FC1-Layer
        outputs = self.fc1(features)

        # Activation layer
        # outputs = self.fc_sigmoid(outputs)


        return outputs



# Tests
# with torch.no_grad():
    # vgg = VGG16(channels=3, height=64, width=64, nr_classes=1)
    # resnet = ResNet50(channels=3, height=64, width=64, nr_classes=1)
    # densenet = DenseNet121(channels=3, height=64, width=64, nr_classes=1)
    # aux = torch.randn(1, 3, 64, 64)
    # out = vgg(aux)
    # out = resnet(aux)
    # out = densenet(aux)

# print(out.shape)