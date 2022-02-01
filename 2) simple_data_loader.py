import torch
from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):
    def __init__(self, text, label):
        self.text = text
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        text = self.text[idx]
        label = self.label[idx]
        sample = {"Text" : text, "Label" : label}
        return sample

text = ['Lotte', 'KT', 'Doosan', 'Samsung', 'LG']
label = ['Busan', 'Suwon', 'Seoul', 'Daegu', 'Seoul']
Dataset = Dataset(text, label)

DataLoader = DataLoader(Dataset, batch_size = 2, shuffle = True)
'''
next(iter(DataLoader))
for dataset in DataLoader:
    print(dataset)
'''