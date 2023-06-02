with open('high_confidence_names_0.7.txt', 'r') as f:
    names = f.readlines()

print(len(names))
print(len(set(names)))