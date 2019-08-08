import numpy as np
import torch
import torch.utils.data
from .vocab import Vocab


class Dataset(torch.utils.data.Dataset):
    def __init__(self, shuffle=True):
        super().__init__()
        self.train_src = []  # [['a', 'b'], ['c', 'd']]
        self.train_tgt = []
        self.test_src = []
        self.test_tgt = []
        self.src_sizes = []
        self.tgt_size = 0
        self.shuffle = shuffle
        self.src_vocab = Vocab(add_pad=True)
        self.tgt_vocab = Vocab(add_pad=False)

    def __len__(self):
        return len(self.train_src)

    def __getitem__(self, i):
        x = self.train_src[i]
        # y = self.train_tgt[i]
        y = self.tgt_vocab.str_to_indices(self.train_tgt[i])
        return {
            'id': i,
            'source': x,
            'target': y
        }

    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    def load_from_file(self, src_path, tgt_path):
        src_file = open(src_path, 'r', encoding='utf-8').readlines()
        for line in src_file:
            self.src_sizes.append(len(line.strip().split()))
            self.train_src.append(line.strip().split())
        tgt_file = open(tgt_path, 'r', encoding='utf-8').readlines()
        for line in tgt_file:
            self.train_tgt.append(line)
        self.src_sizes = np.array(self.src_sizes)

    def load_test_file(self, test_src, test_tgt):
        test_src_file = open(test_src, 'r', encoding='utf-8').readlines()
        for line in test_src_file:
            self.test_src.append(line.strip().split())
        test_tgt_file = open(test_tgt, 'r', encoding='utf-8').readlines()
        for line in test_tgt_file:
            self.test_tgt.append(line)

    def build_vocab(self):
        all_src = self.train_src + self.test_src
        with open('src.vocab', 'w+', encoding='utf-8') as f:
            num = 1
            for sent in all_src:
                for word in sent:
                    if word not in self.src_vocab.wtoi:
                        self.src_vocab.add_word(word)
                        f.writelines(word+' '*2+str(num) + '\n')
                        num += 1
        all_tgt = self.train_tgt + self.test_tgt
        num = 0
        with open('tgt.vocab', 'w+', encoding='utf-8') as f:
            for label in all_tgt:
                if label not in self.tgt_vocab.wtoi:
                    self.tgt_vocab.add_word(label)
                    f.writelines(label.strip()+ ' ' * 2 + str(num) + '\n')
                    num += 1
        self.tgt_size = len(self.tgt_vocab)

    def get_test(self):
        src_dic = {}
        test_src = [self.src_vocab.str_to_indices(sample) for sample in self.test_src]
        src_sizes = torch.LongTensor([len(s) for s in test_src])
        max_len = max([len(i) for i in test_src])
        test_src = torch.LongTensor(self.padding(test_src, max_len))
        src_lengths, sort_order = src_sizes.sort(descending=True)
        final_src = test_src.index_select(0, sort_order)
        test_tgt = torch.LongTensor([self.tgt_vocab.str_to_indices(label) for label in self.test_tgt])
        test_tgt = test_tgt.index_select(0, sort_order)
        src_dic['src_tokens'] = final_src
        src_dic['src_lengths'] = src_lengths
        return (src_dic, test_tgt)

    def padding(self, batch_indice, max_length):
        padding_idx = self.src_vocab.pad()
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




