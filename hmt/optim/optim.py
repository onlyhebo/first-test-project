import torch.optim as optim
class Optim():
    def __init__(self, model):
        self.optimizer = optim.SGD(model.parameters(), lr = 0.1)

    def step(self):
        self.optimizer.step()
