with open('unlabelled_imgs.txt', 'r') as f:
    lines = f.readlines()
    # '/kaggle/input/inf473v-2023-challenge-v2/compressed_dataset/unlabelled/004ZzbhY9LKHQx9.jpg'
    lines = ['/content/unlabelled/' + line.replace('/kaggle/input/inf473v-2023-challenge-v2/compressed_dataset/unlabelled/', '')  for line in lines]

with open('unlabelled_imgs_content.txt', 'w') as f:
    f.writelines(lines)