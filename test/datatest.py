import torch
class dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.randn(10, 2)
        self.label = torch.randn(1)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i], self.label[i]
data = dataset()
itr = torch.utils.DataLoader(dataset=data, batch_size = 2)
for data in itr:
    print(data)
