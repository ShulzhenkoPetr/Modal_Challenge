import argparse
import datetime
import os
import json
import csv
import numpy as np
import pandas as pd

from transformers import ViTMAEModel, AutoImageProcessor, ViTForImageClassification

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import tqdm

from PIL import Image


class OneImageFolder(Dataset):
    def __init__(self, txt_path, transform=None, hugging_mae=False):
        with open(txt_path, 'r') as f:
            self.files = sorted(f.readlines())

        with open('Modal_Challenge/labels_dict.json', 'r') as f:
            self.labels_dict = json.load(f)

        self.transform = transform
        self.hugging_mae = hugging_mae

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = Image.open(img_path.rstrip('\n')).convert('RGB')

        img_name = os.path.splitext(img_path)[0].replace(
            '../gdrive/MyDrive/Modal_Challendge_dataset/compressed_dataset/test/', '')

        # img_name = img_path.replace('../gdrive/MyDrive/Modal_Challendge_dataset/compressed_dataset/test/', '')

        if self.transform:
            img = self.transform(img)

        return img, img_name

    def __len__(self):
        return len(self.files)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_transforms = transforms.Compose([
        transforms.Resize(args.input_size, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    dataset_test = OneImageFolder(
        args.data_path_test,
        transform=test_transforms,
        hugging_mae=args.hugging_mae
    )
    print(len(dataset_test))

    class_names_digit_dict = dataset_test.labels_dict

    digit_class_names_dict = {dig: name for name, dig in class_names_digit_dict.items()}

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    if args.hugging_mae:
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        model.classifier.out_features = 48
        model.classifier.weight = torch.nn.Parameter(model.classifier.weight[:48, :])
        model.classifier.bias = torch.nn.Parameter(torch.randn(48))

        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])

    model.to(device)

    confident_names = []
    submission = pd.DataFrame(columns=["id", "label"])

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():

        model.eval()

        for batch in tqdm.tqdm(data_loader_test):
            images = batch[0]
            names = batch[-1]
            images = images.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                output = model(images)
                output_logits = output.logits

            label_digits = output_logits.argmax(dim=1)

            predictions = [digit_class_names_dict[pred] for pred in label_digits.cpu().numpy()]

            submission = pd.concat(
                [
                    submission,
                    pd.DataFrame({"id": names, "label": predictions}),
                ]
            )

            # Sort samples with strong confidence
            # names = np.array(names)
            # output_logits = softmax(output_logits)
            # confidence = output_logits.max(dim=1).values.cpu().numpy()
            # high_confidence_names = names[confidence > 0.7]

            # confident_names.extend(high_confidence_names)

    submission.to_csv(args.output_dir + '/submission.csv', index=False)

    # with open(args.output_dir + '/high_confidence_names.txt', 'w') as f:
    #     f.write("".join(confident_names))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ViT / MAE prediction', add_help=False)
    parser.add_argument('--data_path_test', default='', help='path to test data txt')
    parser.add_argument('--hugging_mae', action='store_true')
    parser.add_argument('--resume', default='',
                        help='path to checkpoint .pth')
    parser.add_argument('--output_dir', default='')

    parser.add_argument('--input_size', default=224)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--num_workers', default=10)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    args = parser.parse_args()

    main(args)