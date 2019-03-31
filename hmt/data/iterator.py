import torch

def collate_fn(batch):
    return batch


class Iterator():

    def __init__(self, dataset):
        self.dataset = dataset

    def batch_by_size(self):
        batch = []
        for i in range(len(self.dataset)):
            if len(batch)>2:
                print(batch)
                yield batch
                batch = batch[2:]
            batch.append(i)
        if len(batch) > 0:
            yield batch
    def get_batch_iterator(self):
        batch_sampler = self.batch_by_size()
        return torch.utils.data.DataLoader(dataset=self.dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

