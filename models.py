import torchvision
import torch
import torch.nn as nn


class ResNetFinetune(nn.Module):
    def __init__(self, num_classes, weights_path=None, frozen=False):
        super().__init__()
        self.weights_path = weights_path
        if weights_path:
            self.backbone = torchvision.models.resnet50(weights=None)
        else:
            self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x