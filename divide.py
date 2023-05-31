import numpy as np

def train_val_div():
    with open('train_colab.txt', 'r') as f:
        lines = f.readlines()

    n = len(lines)

    vals = []
    for i in range(48):
        random_indx = np.random.choice(15, 3, replace=False)
        for j in range(3):
            vals += [lines[i * 15 + random_indx[j]]]

    for i in range(len(vals)):
        lines.remove(vals[i])

    with open('train_train.txt', 'w') as f:
        f.writelines(lines)

    with open('train_val.txt', 'w') as f:
        f.writelines(vals)

def imagenet_div():
    with open('high_confidence_names.txt', 'r') as f:
        names = f.readlines()

    with open('unlabelled_imgs_content.txt', 'r') as f:
        all_paths = f.readlines()

    not_imagenet_paths = [path
                          for path in all_paths
                          if path.replace('/content/unlabelled/', '') not in names]

    with open('unlabelled_not_imagenet_content.txt', 'w') as f:
        f.writelines(not_imagenet_paths)


if __name__ == '__main__':
    imagenet_div()