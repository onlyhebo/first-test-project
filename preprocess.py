import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--src-path', default='./data/src.txt', type=str, help='src path')
parser.add_argument('--tgt-path', default='./data/tgt.txt', type=str, help='tgt path')
parser.add_argument('--split-per', default=0.8, type=float, help='train test split')
parser.add_argument('--shuffle', default=True, help='shuffle data')
opt = parser.parse_args()

src = open(opt.src_path, 'r', encoding='utf-8').readlines()
tgt = open(opt.tgt_path, 'r', encoding='utf-8').readlines()

train_src = []
train_tgt = []
test_src = []
test_tgt = []

class_dic = {}
for i in range(len(tgt)):
    if tgt[i] in class_dic:
        class_dic[tgt[i]].append(src[i])
    else:
        class_dic[tgt[i]] = [src[i]]

for key in class_dic:
    total_num = len(class_dic[key])
    train_num = int(total_num*opt.split_per)
    if opt.shuffle:
        indices = np.random.permutation(total_num)
    train_idx = indices[:train_num]
    test_idx = indices[train_num:]
    for idx in train_idx:
        train_src.append(class_dic[key][idx])
        train_tgt.append(key)
    for idx in test_idx:
        test_src.append(class_dic[key][idx])
        test_tgt.append(key)

with open('./data/train_src.txt', 'w+', encoding='utf-8') as f:
    for line in train_src:
        f.writelines(line)

with open('./data/train_tgt.txt', 'w+', encoding='utf-8') as f:
    for line in train_tgt:
        f.writelines(line)

with open('./data/test_src.txt', 'w+', encoding='utf-8') as f:
    for line in test_src:
        f.writelines(line)

with open('./data/test_tgt.txt', 'w+', encoding='utf-8') as f:
    for line in test_tgt:
        f.writelines(line)

