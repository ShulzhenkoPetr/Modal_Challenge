import os
import argparse
import numpy as np
import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable

from models import ResNetFinetune

from utils.Dataset import ModalDataset

def create_data_loader(path: str, mode: str, indices: list, batch_size: int, n_cpu: int):
    """
    Creates a train data loader
    :param path: path to images
    :param mode: train_cross_val or val
    :param indices: indices for validation in cross validation
    :param batch_size: Size of a batch
    :param n_cpu: number of workers cpu
    :return torch DataLoader
    """

    dataset = ModalDataset(path, mode)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu)
    return dataloader

def len_dataset(path: str) -> int:
    """
    Finds the length of required dataset (.txt file)
    :param path: path to a txt file
    :return: length
    """
    with open(path, 'r') as f:
        return len(f.readlines())


def evaluate(model, val_dataloader: DataLoader) -> tuple:
    """
    Evaluates the model on a val set
    :param model: model
    :param val_dataloader: val dataloader
    :return: tuple = (test_loss, test_accuracy)
    """
    model.eval()

    losses = []
    accuracy = []

    loss_fn = torch.nn.CrossEntropyLoss()

    for imgs, target in val_dataloader:
        #imgs = Variable(imgs.to(device, non_blocking=True), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            loss = loss_fn(outputs)

            acc = torch.sum(outputs.detach().cpu().numpy().argmax(dim=1) == target) / len(target)

            losses.append(loss.detach().cpu())
            accuracy.append(acc)


    return np.mean(losses), np.mean(accuracy)



def train():
    parser = argparse.ArgumentParser(description="Baseline Model training")
    parser.add_argument("-d", "--data", type=str, default="", help="Path to root data folder")
    parser.add_argument("--nb_classes", type=int, default=48, help="Number of classes")
    parser.add_argument("--k_folds", type=int, default=3, help="Number of folds in cross-validation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=int, default=0.001, help="Learning rate")
    parser.add_argument("--decay", type=int, default=0.0005, help="Adam decay")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--pretrained_weights", type=str,
                        help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=1,
                        help="Interval of epochs between saving model weights")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory in which the checkpoints are stored")
    parser.add_argument("--evaluation_interval", type=int, default=1,
                        help="Interval of epochs between evaluations on validation set")
    parser.add_argument("--softmax_thres", type=float, default=0.5,
                        help="Threshold for softmax confidence before adding a label to an unlabeled image")
    parser.add_argument("--logdir", type=str, default="logs",
                        help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # logger = Logger(args.logdir)  # Tensorboard logger

    # Create output directories if missing
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Lists accumulating cross validation metrics
    losses = []
    accuracy = []

    #Choose images for each fold of cross validation
    train_len = len_dataset(args.data + '/train_imgs.txt')
    indices = np.random.choice(train_len, size=(args.k_folds, int((1 / args.k_folds) * train_len)), replace=False)

    val_accuracy = []
    val_loss = []

    for i_fold in range(args.k_folds):

        print(f"\n---- Training Model {i_fold} out of {args.k_folds}----")

        train_dataloader = create_data_loader(
            args.data,
            'train',
            indices[i_fold],
            args.batch_size,
            args.n_cpu)
        val_dataloader = create_data_loader(
            args.data,
            'val',
            indices[i_fold],
            args.batch_size,
            args.n_cpu)

        model = ResNetFinetune(args.nb_classes, frozen=True)

        params = [p for p in model.parameters() if p.requires_grad]

        optimizer = optim.Adam(
            params,
            lr = args.lr,
            weight_decay = args.decay)

        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(1, args.epochs + 1):
            model.train()

            for batch_i, (_, imgs, target) in enumerate(tqdm.tqdm(train_dataloader, desc=f"Training Epoch {epoch}")):

                imgs = Variable(imgs.to(device, non_blocking=True))

                outputs = model(imgs)
                loss = loss_fn(outputs)
                loss.backward()

                #change lr ... ? maybe scheduler

                optimizer.step()
                optimizer.zero_grad()

                #logging ?

            #Evaluate
            if epoch == args.epochs:
                print("\n---- Evaluating Model ----")
                # Evaluate the model on the validation set
                metrics_output = evaluate(
                    model,
                    val_dataloader
                )
                val_loss[i_fold] = metrics_output[0]
                val_accuracy[i_fold] = metrics_output[1]




if __name__ == '__main__':
    train()