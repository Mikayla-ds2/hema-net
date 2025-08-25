import torch
import torch.nn as nn
import torchvision.models as models

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 8)  # 8 classes in BloodMNIST

    def forward(self, x):
        return self.resnet(x)