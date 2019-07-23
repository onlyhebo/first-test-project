from torch import nn
import torch.distributed as dist
import torch


class Trainer:
    def __init__(self, model, optim, opt):
        self.model = model
        self.model.train()
        self.criterion = nn.CrossEntropyLoss()
        self.optim = optim
        self.opt = opt
        self.step = 0

    def average_gradients(self):
        for param in self.model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= 2

    def train(self, batch, test_data, epoch):
        normalization = len(batch['net_input'])
        out = self.model(batch['net_input'])
        loss = self.criterion(out, batch['target'])
        loss.div(normalization).backward()
        if self.opt.num_process > 1:
            self.average_gradients()
        # print(loss)
        self.optim.clip_grad_norm(self.opt.clip_norm)
        self.optim.step()
        self.optim.zero_grad()
        self.step += 1
        if self.step % 50 == 0:
            with torch.set_grad_enabled(False):
                self.valid(test_data, epoch, loss)

    def valid(self, test_data, epoch, train_loss):
        self.model.eval()
        net_input = test_data[0]
        target = test_data[1]
        out = self.model(net_input)
        pred = torch.argmax(out, dim=1)
        acc = int(torch.sum(pred.eq(target)))/pred.size()[0]
        print('Epoch is {}. Train loss is {}. Test accuracy is {}'.format(epoch, train_loss, acc))
        self.model.train()


