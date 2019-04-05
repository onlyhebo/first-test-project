import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):

    def __init__(self, shuffle = True):
        super().__init__()
        # self.data = torch.randn(100,2)
        # self.target = torch.randn(100)
        self.src = []
        self.tgt = []
        self.src_sizes = []
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.src[i]
        y = self.tgt[i]
        return x, y

    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    def load_from_file(self, src_path, tgt_path):
        src_file = open(src_path, 'r', encoding='utf-8').readlines()
        # file = pd.read_excel(path)
        for line in src_file:
            self.src_sizes.append(len(line))
            self.src.append(line.strip().split())
        tgt_file = open(tgt_path, 'r', encoding='utf-8').readlines()
        for line in tgt_file:
            self.tgt.append(line.strip())


