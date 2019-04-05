from torch import nn
import torch.distributed as dist
from data import Dataset


def collate_fn(x):
    return x


class Trainer():
    def __init__(self, model, optim, opt):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optim = optim
        self.opt = opt

    def average_gradients(self):
        for param in self.model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= 2

    def train(self, batch):
        out = self.model(batch[0])
        loss = self.criterion(out, batch[1])
        loss.backward()
        self.average_gradients()
        print(loss)
        self.optim.step()

    def build_dataset(self, opt):
        dataset = Dataset(shuffle=True)
        dataset.load_from_file(opt.src_file_path, opt.tgt_file_path)
        return dataset


