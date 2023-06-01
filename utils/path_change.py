with open('train_val.txt', 'r') as f:
    lines = f.readlines()
    # '/kaggle/input/inf473v-2023-challenge-v2/compressed_dataset/unlabelled/004ZzbhY9LKHQx9.jpg'
    lines = ['/content/gdrive/MyDrive/Modal_Challendge_dataset/compressed_dataset/' + line.replace('/content/gdrive/MyDrive/Colab_Notebooks/Modal/', '')  for line in lines]

with open('train_val_gdrive.txt', 'w') as f:
    f.writelines(lines)