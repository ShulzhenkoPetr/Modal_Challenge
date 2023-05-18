import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torchvision.io import read_image
import torchvision.transforms as T
import numpy as np
from utils.Preprocessing import data_augmentation_normalization_resize, add_random_blocks
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
    def create_img_txt(path: str, mode: str, indexes: list = []) -> None:
        os.chdir(path)
        img = []
        if mode == 'train':
            os.chdir(mode)
            class_list = os.listdir()
            for class_ in tqdm(class_list):
                if not class_.startswith('.'):
                    os.chdir(class_)
                    cur = os.listdir()
                    img += [path + '/train/' + class_ + '/' + cur[i] for i in range(len(cur))]
                    os.chdir('..')

            with open(path + '/train_imgs.txt', 'w') as f:
                for i in range(len(img)):
                    f.write(img[i] + '\n')

        if mode == 'test':
            os.chdir(mode)
            images = os.listdir()
            img += images
            img = [path + '/test/' + img[i] for i in range(len(img))]

            with open(path + '/test_imgs.txt', 'w') as f:
                for i in range(len(img)):
                    f.write(img[i] + '\n')

        if mode == 'unlabelled':
            os.chdir(mode)
            images = os.listdir()
            img += images
            img = [path + '/unlabelled/' + img[i] for i in range(len(img))]

            with open(path + '/unlabelled_imgs.txt', 'w') as f:
                for i in range(len(img)):
                    f.write(img[i] + '\n')

        if mode == 'train_cross_validation' or mode == 'val_cross_validation':
            os.chdir('train')
            img_val = []
            class_list = os.listdir()
            print(class_list)
            for class_ in tqdm(class_list):

                if not class_.startswith('.'):
                    os.chdir(class_)
                    cur = os.listdir()
                    img += [path + '/train/' + class_ + '/' + cur[i] for i in range(len(cur))]
                    os.chdir('..')

            for i in range(len(indexes)):
                img_val += [img[indexes[i]]]

            if mode == 'train_cross_validation':
                for i in range(len(img_val)):
                    img.remove(img_val[i])

                with open(path + '/train_cross_validation_imgs.txt', 'w') as f:
                    for i in range(len(img)):
                        f.write(img[i] + '\n')

            if mode == 'val_cross_validation':
                with open(path + '/val_cross_validation_imgs.txt', 'w') as f:
                    for i in range(len(img)):
                        f.write(img[i] + '\n')

        os.chdir('..')

    def get_image(self, img_path: str, labels_dict: dict = {}):

        img = read_image(img_path.rstrip('\n'))

        curr = img_path.replace(self.path + '/', '').rstrip('\n').split('/')

        if len(curr) == 3:
            return img, labels_dict[curr[1]]

        else:
            return img

    # def get_labels_dict(self, path: str) -> dict:
    #
    #     print(os.getcwd())
    #
    #     os.chdir(path)
    #
    #     labels = os.listdir()
    #     labels_dict = {}
    #     for i in range(len(labels)):
    #         labels_dict.update({labels[i]: i})
    #     os.chdir('..')
    #     return labels_dict

    def __init__(self, mode: str, path: str, indexes: list = [], img_size: tuple = (224, 224)) -> None:

        self.img_size = img_size
        self.path = path
        self.mode = mode
        self.file_name = mode + '_imgs.txt'

        with open(path + '/labels_dict.json', 'r') as f:
            self.labels_dict = json.load(f)

        #self.create_img_txt(path, mode, indexes=indexes)

    def __len__(self) -> int:
        with open(self.path + '/' + self.file_name, 'r') as f:
            return len(f.readlines())

    def __getitem__(self, idx) -> dict:

        if mode == 'train' or mode == 'train_cross_validation':
            with open(self.path + '/' + self.file_name, 'r') as f:
                curr_imgs = f.readlines()
            img, target = self.get_image(curr_imgs[idx], self.labels_dict)
            img = T.Resize(self.img_size)(img)
            # print(data_augmentation_normalization_resize(img))
            aug_imgs = torch.stack(data_augmentation_normalization_resize(img))
            aug_imgs = torch.cat((aug_imgs, add_random_blocks(aug_imgs)))
            index = np.random.randint(0, len(self.labels_dict))
            img = aug_imgs[index].to(float) / 255.
            return {'image': img, 'target': target}


        if mode == 'val_cross_validation':
            with open(self.path + '/' + self.file_name, 'r') as f:
                curr_imgs = f.readlines()
            img, target = self.get_image(curr_imgs[idx], self.labels_dict)
            img = T.Resize(self.img_size)(img)
            return {'image': img.to(float) / 255., 'target': target}

        if mode == 'test' or mode == 'unlabelled':
            with open(self.path + '/' + self.file_name, 'r') as f:
                img = self.get_image(f.readlines()[idx])
                img = T.Resize(self.img_size)(img)
            return {'image': img.to(float) / 255.}


if __name__ == '__main__':
    mode = 'train'
    indexes = [0, 5, 4]
    path = '/Users/sanek_tarasov/Documents/École polytechnique/2A/P3/Modal'
    dataset = ModalDataset(mode, path, indexes=indexes)
    # a = '/Users/sanek_tarasov/Documents/École polytechnique/2A/P3/Modal/test/7hHUDQZ86TVerFY.jpg\n'
    print(show_tensor_images(dataset.__getitem__(5)['image'], num_images=1))
    # a.rstrip('\n')
    # print(a)
