import torchvision
import torch
import torch.nn as nn
from tqdm import tqdm
from Dataset import ModalDataset, show_tensor_images
from torchsummary import summary


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        return x

if __name__ == '__main__':

    path = '/Users/sanek_tarasov/Documents/École polytechnique/2A/P3/Modal'
    train_dataset = ModalDataset('train', path)
    ex = train_dataset[0][0]
    # show_tensor_images(ex, num_images=1)
    descr = ResNet()
    print(descr(ex.float()[None, ...]).shape)
    # print(summary(descr.backbone, (3, 224, 224)))
