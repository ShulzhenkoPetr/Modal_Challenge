import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torchvision.io import read_image
import torchvision.transforms as T
import numpy as np
import os
import json
import matplotlib.pyplot as plt


def show_tensor_images(image_tensor: torch.Tensor, num_images: int = 25, size: tuple = (3, 224, 224)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


class ModalDataset(Dataset):
    @staticmethod
    def create_img_txt(path: str, mode: str, indexes: list = None) -> None:

        if mode == 'train_cross_validation' or mode == 'val_cross_validation':
            imgs_val = []

            with open(path + '/train_imgs.txt', 'r') as f:
                imgs = f.readlines()

            for i in range(len(indexes)):
                imgs_val += [imgs[indexes[i]]]

            if mode == 'train_cross_validation':
                for i in range(len(imgs_val)):
                    imgs.remove(imgs_val[i])

                with open(path + '/train_cross_validation_imgs.txt', 'w') as f:
                    for i in range(len(imgs)):
                        f.write(imgs[i])

            if mode == 'val_cross_validation':
                with open(path + '/val_cross_validation_imgs.txt', 'w') as f:
                    for i in range(len(imgs_val)):
                        f.write(imgs_val[i])

    def get_image(self, img_path: str, labels_dict: dict = None):

        img = read_image(img_path.rstrip('\n'))

        curr = img_path.replace('/Users/sanek_tarasov/Downloads/compressed_dataset/', '').rstrip('\n').split('/')
        if len(curr) == 3:
            return img, labels_dict[curr[1]]

        else:
            return img

    def __init__(self, mode: str, path: str, indexes: list = [], img_size: tuple = (224, 224)) -> None:

        self.img_size = img_size
        self.path = path
        self.mode = mode
        self.file_name = mode + '_imgs.txt'

        with open(self.path + '/labels_dict.json', 'r') as f:
            self.labels_dict = json.load(f)

        self.create_img_txt(path, mode, indexes=indexes)

    def __len__(self) -> int:
        with open(self.path + '/' + self.file_name, 'r') as f:
            return len(f.readlines())

    def __getitem__(self, idx) -> tuple:

        if self.mode == 'train':
            with open(self.path + '/' + self.file_name, 'r') as f:
                curr_imgs = f.readlines()
            img, target = self.get_image(curr_imgs[idx], self.labels_dict)
            img = T.Resize(self.img_size, antialias=None)(img)
            return img.to(float) / 255., target

        if self.mode == 'test' or self.mode == 'unlabelled':
            with open(self.path + '/' + self.file_name, 'r') as f:
                img = self.get_image(f.readlines()[idx])
                img = T.Resize(self.img_size, antialias=None)(img)
            return img.to(float) / 255.


if __name__ == '__main__':
    mode = 'train'
    indexes = [0, 5, 4]
    path = '/Users/sanek_tarasov/Documents/École polytechnique/2A/P3/Modal'
    dataset = ModalDataset(mode, path, indexes=indexes)
    # a = '/Users/sanek_tarasov/Documents/École polytechnique/2A/P3/Modal/test/7hHUDQZ86TVerFY.jpg\n'
    print(show_tensor_images(dataset.__getitem__(5)['image'], num_images=1))
    # a.rstrip('\n')
    # print(a)
