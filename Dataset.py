import torch
from tqdm.notebook import tqdm
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torchvision.io import read_image
import numpy as np
import os
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt


def show_tensor_images(image_tensor: torch.Tensor, num_images: int = 25, size: tuple = (3, 224, 224)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


class ModalDataset(Dataset):
    @staticmethod
    def create_img_txt(path: str, num_img_in_class: int = 15, num_val: int = 4) -> None:
        os.chdir(path)
        img = []
        if path == 'train':
            img_val = []
            class_list = os.listdir()
            for class_ in tqdm(class_list):
                os.chdir(class_)

                cur = os.listdir()
                img += ['train/' + class_ + '/' + cur[i] for i in range(len(cur))]
                os.chdir('..')

            # for i in range(len(class_list)):
            #     random_indx = np.random.choice(num_img_in_class, num_val, replace=False)
            #     for j in range(num_val):
            #         img_val += [img[i * num_img_in_class + random_indx[j]]]

            # for i in range(len(img_val)):
            #     img.remove(img_val[i])

        else:
            img += os.listdir()
            img = [path + '/' + img[i] for i in range(len(img))]
        os.chdir('..')

        # if path == 'train':
        #     with open('val_imgs.txt', 'w') as f:
        #         for i in range(len(img_val)):
        #             f.write(img_val[i] + '\n')

        with open(path + '_imgs.txt', 'w') as f:
            for i in range(len(img)):
                f.write(img[i] + '\n')

    @staticmethod
    def get_image(img_path: str, labels_dict) -> tuple:

        curr = img_path.rstrip('\n').split('/')

        if len(curr) == 2:
            os.chdir(curr[0])
            img = read_image(curr[1])
            os.chdir('..')
            return img
        else:
            os.chdir(curr[0])
            os.chdir(curr[1])
            img = read_image(curr[2])
            os.chdir('..')
            os.chdir('..')
            return img, labels_dict[curr[1]]

    def get_labels_dict(self, path: str) -> dict:

        os.chdir(path)

        labels = os.listdir()
        labels_dict = {}
        for i in range(len(labels)):
            labels_dict.update({labels[i]: i})
        os.chdir('..')
        return labels_dict

    def __init__(self, mode: bool, num_img_in_class: int = 15, num_val: int = 4) -> None:

        self.labels_dict = self.get_labels_dict('train')
        self.train_imgs_file = 'train_imgs.txt'
        # self.val_imgs_file = 'val_imgs.txt'
        self.test_imgs_file = 'test_imgs.txt'
        self.unlab_train_img = 'unlabelled_imgs.txt'
        self.mode = mode
        self.create_img_txt('train')
        self.create_img_txt('test')
        # self.create_img_txt('unlabelled')

    def __len__(self) -> int:
        if self.mode == 'train':
            return
        return self.num_img

    def __getitem__(self, idx) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return 0
