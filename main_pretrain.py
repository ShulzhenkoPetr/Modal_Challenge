# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

from transformers import ViTMAEForPreTraining, AutoImageProcessor

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch

import glob
from PIL import Image
from torch.utils.data import Dataset


class OneImageFolder(Dataset):
    def __init__(self, txt_path, transform=None):
        with open(txt_path, 'r') as f:
            self.files = sorted(f.readlines())

        self.transform = transform

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = Image.open(img_path.rstrip('\n')).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.files)


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--pretrained_encoder', default='',
                        help='path to pretrained ViT encoder weights')
    parser.add_argument('--hugging_mae', action='store_true',
                        help='Hugging Face pretrained encoder-decoder model')


    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    if args.hugging_mae:
        transform_train = None
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset_train = OneImageFolder(args.data_path, transform=transform_train)
    print(len(dataset_train))

    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if args.log_dir is not None:
        log_dir = os.path.join(
            args.log_dir,
            args.model,
            datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        logger = SummaryWriter(log_dir)
    else:
        logger = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    if args.hugging_mae:
        if args.resume:
            model = ViTMAEForPreTraining.from_pretrained(args.resume)
        else:
            model = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')
    else:
        model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    # model.to(device)
    #
    # model_without_ddp = model
    print("Model = %s" % str(model))

    # eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    eff_batch_size = args.batch_size * args.accum_iter * 1

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #     model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()


    #Load model:
    #load weights from checkpoint
    if not args.hugging_mae:
        misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

    # Unfreeze encoder layers ..

    #Load pretrained frozen encoder
    if args.pretrained_encoder:
        encoder_weights = torch.load(args.pretrained_encoder, map_location='cpu')
        encoder_layers = list(encoder_weights['model'].keys())

        sd = model.state_dict()

        with torch.no_grad():
            for layer in sd:
                if layer in encoder_layers:
                    sd[layer].data = encoder_weights['model'][layer].data

        model.load_state_dict(sd)

        #Freeze encoder layers
        mae_layers = list(model.state_dict().keys())
        for i, param in enumerate(model.parameters()):
            if mae_layers[i] in encoder_layers:
                param.requires_grad_(False)

    model.to(device)

    #Training loop:
    if args.hugging_mae:
        image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    else:
        image_processor = None

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        if args.hugging_mae:
            rnd_visual_samples = image_processor(
                images=[dataset_train[10], dataset_train[1000], dataset_train[10000]],
                return_tensors="pt"
            )
        else:
            rnd_visual_samples = torch.stack((dataset_train[10], dataset_train[1000], dataset_train[10000])).to(device)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=logger,
            args=args,
            image_processor=image_processor,
            rnd_visual_samples=rnd_visual_samples
        )
        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if logger is not None:
                logger.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)