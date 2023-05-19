from Dataset import show_tensor_images, ModalDataset
from Resnet import ResNet
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader


def image_embeddings(image_descriptor, dataset, emd_dim: int = 2048) -> torch.Tensor:
    '''

    :param image_descriptor: image descriptor which will give us the images embeddings
    :param dataset: images
    :return: torch.Tensor size of [nb_images, emb_dim]
    '''
    emd = torch.zeros(len(dataset), emd_dim)
    targets = torch.zeros(len(dataset))
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    if dataset.mode == 'train':
        for i, (img, target) in enumerate(tqdm(loader, desc='Images processing')):
            emd[i * loader.batch_size: (i + 1) * loader.batch_size, :] = image_descriptor(img.float())
            targets[i * loader.batch_size: (i + 1) * loader.batch_size] = target
        return emd, targets
    else:
        for i, img in enumerate(tqdm(loader, desc='Images processing')):
            emd[i * loader.batch_size: (i + 1) * loader.batch_size, :] = image_descriptor(img.float())
        return emd


if __name__ == '__main__':
    path = '/Users/sanek_tarasov/Documents/École polytechnique/2A/P3/Modal'
    train_dataset = ModalDataset('train', path)
    unlabeled_dataset = ModalDataset('unlabelled', path)
    descr = ResNet()

    image_embeddings = image_embeddings(descr, unlabeled_dataset)
    # print(image_embeddings.shape, target.shape)
    torch.save(image_embeddings, 'images_emd_unlabelled.pt')
    # torch.save(target, 'target.pt')
