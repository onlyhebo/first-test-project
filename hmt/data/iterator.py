import torch
import numpy as np

def collate_fn(batch):
    return batch


class Iterator():

    def __init__(self, dataset, opt):
        self.dataset = dataset
        self.opt = opt

    def is_batch_full(self, batch_len):
        if batch_len == opt.batch_size:
            return True
        return False

    def batch_by_size(self, indices):
        batch = []
        for idx in indices:
            batch_len = len(batch)
            if self.is_batch_full(batch_len):
                yield batch[:self.opt.batch_size]
                batch = batch[self.opt.batch_size:]
            batch.append(idx)
        if len(batch>0):
            yield batch

    def get_batch_iterator(self):
        with np.random.seed(self.opt.seed):
            indices = self.dataset.ordered_indices()
        batch_sampler = self.batch_by_size(indices)
        return torch.utils.data.DataLoader(dataset=self.dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

