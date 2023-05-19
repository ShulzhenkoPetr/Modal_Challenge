import faiss
# from utils.Resnet import ResNet
# from utils.Preprocessing import image_embeddings
# from utils.Dataset import ModalDataset, show_tensor_images
from tqdm import tqdm
import torch


if __name__ == '__main__':

    images_emd = torch.load('utils/images_emd.pt')
    target = torch.load('utils/target.pt')
    print(images_emd)