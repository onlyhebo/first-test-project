from hmt.model import Test_model
from hmt.optim import Optim
from hmt.trainer import Trainer
import torch
import torch.utils.data
import multiprocessing as mp
import torch.distributed as dist
from hmt.data import DataDistributor
import os
from torch.multiprocessing import Queue
from hmt.data import iterator
from hmt.multiprocessing_pdb import pdb

class Dataset(torch.utils.data.Dataset):

    def __init__(self):
        super().__init__()
        self.data = torch.randn(100,2)
        self.target = torch.randn(100)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i]
        y = self.target[i]
        return x,y

def single_process(data_queue, rank):
    if rank == 0:
        pdb.set_trace()
    model = Test_model()
    optim = Optim(model)
    trainer = Trainer(model, optim)
    for _ in range(100):
        data = data_queue.get()
        trainer.train(data)

def init_process(fn, data_queue, rank):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '24580'
    dist.init_process_group(backend='gloo', rank = rank, world_size = 2)
    fn(data_queue, rank)

def data_manager(data_distributor):
    dataset = Dataset()
    itr = iterator.Iterator(dataset)
    itr = itr.get_batch_iterator()
    for batch in itr:
        data_distributor.send_example_list(batch)

def main():
   # mp.set_start_method('spawn')
    processes = []
    error_queue = mp.SimpleQueue()
    data_distributor = DataDistributor()
    for rank in range(2):
        data_queue = Queue()
        data_distributor.add(data_queue)
        p = mp.Process(target = init_process, args=(single_process, data_queue, rank))
        p.start()
        processes.append(p)
    for i in range(15):
        data_manager(data_distributor)
    for p in processes:
        p.join()
if __name__ == '__main__':
    main()
