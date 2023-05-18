import torch
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
from torchvision.io import read_image
import torchvision.transforms as T
import numpy as np
import os
from typing import List, Dict
import matplotlib.pyplot as plt


def show_tensor_images(image_tensor: torch.Tensor, num_images: int = 25, size: tuple = (3, 768, 768)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def get_images(path: str) -> tuple((Dict, List, List)):
    os.chdir(path)

    labels = os.listdir()
    labels.remove('.DS_Store')
    labels_dict = {}
    for i in tqdm(range(len(labels))):
        labels_dict.update({i: labels[i]})

    data = []
    target = []

    for label_number, label in labels_dict.items():

        os.chdir(label)
        for img_path in os.listdir():
            img = read_image(img_path)
            data += [img]
            target += [label_number]
        os.chdir('..')
    os.chdir('..')
    return labels_dict, data, target


def data_augmentation_normalization_resize(x: torch.Tensor, target: List = [], rotation: bool = True,
                                           num_rotation: int = 6,
                                           gaussian_noise: bool = True, noise_factor: float = 0.6, flips: bool = True,
                                           normalization: bool = True, resize: bool = True,
                                           size: tuple = (224, 224)) -> tuple((torch.Tensor, torch.Tensor)):
    aug_data = []
    aug_target = []

    # for i in tqdm(range(len(x))):
    # aug_target += [target[i]]
    # img = x[i]

    # if resize:
    #     img = T.Resize(size)(x[i])
    #     aug_data += [img]
    # else:
    #     aug_data += [x[i]]

    img = x.clone()

    if rotation:
        rot_imgs = [T.RandomRotation(degrees=angle)(img) for angle in
                    range(360 // num_rotation, 360, 360 // num_rotation)]
        # rot_target = [target[i] for i in range(len(rot_imgs))]
        aug_data += rot_imgs
        # aug_target += rot_target

    if gaussian_noise:
        noisy = img.to(float) + torch.randn_like(img.to(float)) * noise_factor

        aug_data += [noisy]
        # aug_target += [target[i]]

    if flips:
        aug_data += [T.RandomHorizontalFlip(p=1)(img)]
        # aug_target += [target[i]]
        aug_data += [T.RandomVerticalFlip(p=1)(img)]
        # aug_target += [target[i]]


    return aug_data

def normalize(x: list) -> torch.Tensor:
    aug_data = torch.stack(x).to(float)
    mean = aug_data.mean(dim=(0, 2, 3))
    std = aug_data.std(dim=(0, 2, 3))
    aug_data = T.Normalize(mean, std)(aug_data)
    return aug_data

def add_random_blocks(img: torch.Tensor, n_k: int = 10, size=32) -> torch.Tensor:
    h, w = size, size
    img_size = img.shape[2]
    # boxes = []
    blocked_img = torch.clone(img)
    for k in range(n_k):
        y, x = np.random.randint(0, img_size - h, 2)
        blocked_img[:, :, y:y + h, x:x + w] = 0
        # boxes.append((x,y,h,w))
    # img = Image.fromarray(img.astype('uint8'), 'RGB')
    return blocked_img


# def save_images(folder_name: str, data: torch.Tensor, target: torch.Tensor, label_dict: dict,
#                 val_proportion: float = .2) -> None:
#     os.mkdir('val_' + folder_name)
#     os.mkdir(folder_name)
#
#     for i in tqdm(range(10)):
#
#         for j in range():


if __name__ == '__main__':
    PATH = 'dataset/train'
    # os.chdir(PATH)
    # print(os.getcwd())
    # os.chdir('gosling')

    labels_dict, data, target = get_images(PATH)
    aug_data, aug_target = data_augmentation_normalization_resize(data, target)
    # print(os.getcwd())
    # save_image(aug_data[0], '001.jpg')
    # show_tensor_images(add_random_blocks(aug_data[9:19, :, :, :]), num_images=9, size=(3, 224, 224))

    dataset = torch.cat((aug_data, add_random_blocks(aug_data)))
    target = torch.cat((aug_target, aug_target))

    num_aug = 9
    folders = set()

    for i in tqdm(range(0, len(target), num_aug)):
        folder = labels_dict[target[i].item()]
        folders.add(folder)
        x = np.random.randint(0, num_aug, 2)
        os.chdir('train_aug')
        if not os.path.exists(folder):
            os.mkdir(folder)
        os.chdir(folder)

        for j in range(num_aug):
            if j != x[0] and j != x[1]:
                save_image(dataset[i + j], f'{j + i}.jpg')
        os.chdir('..')
        os.chdir('..')

        os.chdir('val_aug')
        if not os.path.exists(folder):
            os.mkdir(folder)
        os.chdir(folder)

        save_image(dataset[i + x[0]], f'{i + 1}.jpg')
        save_image(dataset[i + x[1]], f'{i + 2}.jpg')

        os.chdir('..')
        os.chdir('..')
