with open('test_imgs.txt', 'r') as f:
    lines = f.readlines()
    lines = ['../gdrive/MyDrive/Modal_Challendge_dataset/compressed_dataset/' + line.replace('/Users/sanek_tarasov/Documents/École polytechnique/2A/P3/Modal/', '')  for line in lines]

with open('test_imgs.txt', 'w') as f:
    f.writelines(lines)