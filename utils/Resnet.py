import torchvision
import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = torchvision.models.resnet50(pretrained=True)

    def forward(self, x):
        x = self.backbone(x)
        return x

if __name__ == '__main__':

    descr = ResNet()

    img