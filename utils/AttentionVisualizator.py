import torch
from transformers import AutoImageProcessor, ViTMAEForPreTraining

import torchvision.transforms as T

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import requests

## https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb


@torch.no_grad()
def visualize_attentions(attentions, image, input_image, show_att_map=False, threshold_division=3):

    """

    :param attentions: attentions params from model
    :param image: image
    :param input_image: if you want you can use preprocessed image (Tensor)
    :param show_att_map: display attention maps without image
    :param threshold_division: we want to divide the attention values into threshold values and multiply by fixed value
    :return: None, just print the images
    """
    att_mat = torch.stack(list(attentions)).clone().detach()
    att_mat = att_mat.squeeze(1)
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)
    #     print(att_mat.shape)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    grid_size = int(np.sqrt(aug_att_mat.size(-1)))

    for i, v in enumerate(joint_attentions):
        # Attention from the output token to the input space.
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        mask = cv2.resize(mask / mask.max(), image.size)[..., np.newaxis]
        mask_min, mask_max = mask.min(), mask.max()
        thresholds = np.sort(np.linspace(mask_min, mask_max, threshold_division))[::-1]
        value = 0.9
        mask_new = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]))
        for j in range(len(thresholds) - 1):
            value /= 2
            mask_threshold = mask < thresholds[j]
            mask_new += mask * mask_threshold * value

        result = (mask_new * image).astype("uint8")
        if show_att_map:
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 16))
            ax1.set_title('Original')
            ax2.set_title('Attention Map_%d Layer with image' % (i + 1))
            ax3.set_title('Attention Map_%d Layer' % (i + 1))
            _ = ax1.imshow(image)
            _ = ax2.imshow(result)
            _ = ax3.imshow(mask_new)
        else:
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
            ax1.set_title('Original')
            ax2.set_title('Attention Map_%d Layer with image' % (i + 1))
            _ = ax1.imshow(image)
            _ = ax2.imshow(result)


if __name__ == '__main__':
    image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    resizer = T.Resize((224, 224), antialias=None)

    output = model(T.ToTensor()(resizer(image))[None, ...], output_attentions=True)
    attentions = output.attentions

    visualize_attentions(attentions, resizer(image), image, True)