import numpy as np

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