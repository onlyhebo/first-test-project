import torch
import numpy as np


def collate_fn(samples, vocab):
    def padding(batch_indice, max_length):
        padding_idx = vocab.pad()
        batch_pad = []
        for sent in batch_indice:
            pad_num = max_length - len(sent)
            if pad_num > 0:
                sent.extend([padding_idx] * pad_num)
                batch_pad.append(sent)
            else:
                assert len(sent) == max_length
                batch_pad.append(sent)
        return batch_pad
    batch_id = [vocab.str_to_indices(sample['source']) for sample in samples]
    src_sizes = torch.LongTensor([len(s) for s in batch_id])
    max_len = max([len(i) for i in batch_id])
    batch_padded = torch.LongTensor(padding(batch_id, max_len))
    src_lengths, sort_order = src_sizes.sort(descending=True)
    final_src = batch_padded.index_select(0, sort_order)
    final_tgt = torch.LongTensor([sample['target'] for sample in samples])
    final_tgt = final_tgt.index_select(0, sort_order)
    batch = {
        'net_input': {
            'src_tokens': final_src,
            'src_lengths': src_lengths
        },
        'target': final_tgt
    }
    return batch  # batch_size * max_length


class Iterator:
    def __init__(self, dataset, opt):
        self.dataset = dataset
        self.opt = opt

    def is_batch_full(self, batch_len):
        if batch_len == self.opt.batch_size:
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
        if len(batch)>0:
            yield batch

    def collate_data(self, samples):
        return collate_fn(samples, self.dataset.src_vocab)

    def get_batch_iterator(self):
        np.random.seed(self.opt.seed)
        indices = self.dataset.ordered_indices()
        batch_sampler = self.batch_by_size(indices)
        return torch.utils.data.DataLoader(dataset=self.dataset, batch_sampler=batch_sampler, collate_fn=self.collate_data)

