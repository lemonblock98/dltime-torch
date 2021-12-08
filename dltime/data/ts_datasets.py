import torch
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from sktime.datasets import base, load_UCR_UEA_dataset
from collections import Counter


class UCR_UEADataset(Dataset):
    "Torch Datasets for UCR/UEA archive"

    def __init__(self, name, split=None, extract_path="ucr_uea_archive", fixed=True, return_y=True):
        assert split in ["train", "test", None]

        super().__init__()
        self.return_y = return_y

        self.x, self.y = load_UCR_UEA_dataset(name, split=split, return_X_y=True, \
            extract_path=extract_path) # x, y => Dataframe
        
        self.y = np.array(self.y)
        self.y_dict = dict([(y, i) for i, y in enumerate(np.unique(self.y))])
        self.y = [self.y_dict[y] for y in self.y]

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        if self.return_y:
            return torch.tensor(self.x.iloc[index]), torch.tensor(self.y[index])
        else:
            return torch.tensor(self.x.iloc[index])


class TS_TAGDataset(Dataset):
    "Given a specific tsDataset and tag, take out the dataset"
    "dataset: data, label"

    def __init__(self, baseset, label):
        labels = np.unique([baseset[i][1] for i in range(len(baseset))])
        assert label in labels
        super().__init__()
        self.tagset = [baseset[i][0] for i in range(len(baseset)) if baseset[i][1] == label]

    def __len__(self):
        return len(self.tagset)
    
    def __getitem__(self, index):
        return self.tagset[index]


class TS_GENDataset(Dataset):
    "Given a specific tsDataset and tag, take out the dataset"
    "dataset: data, label"

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, index):
        return self.data[index], torch.tensor(self.label)

if __name__ == "__main__":
    # x, y = load_UCR_UEA_dataset("ACSF1", return_X_y=True, extract_path="ucr_uea_archive")
    dataset = UCR_UEADataset("Worms", split="train")
    print(Counter(dataset.y))
    print(len(dataset))
    tagset = TS_TAGDataset(dataset, label=0)
    print(tagset[0].size())

