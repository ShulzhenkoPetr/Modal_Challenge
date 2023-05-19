import torchvision
import torch
import torch.nn as nn


class ResNetFinetune(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = torchvision.models.resnet50(pretrained=True)

    def forward(self, x):
        x = self.backbone(x)
        return x