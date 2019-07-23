from hmt.model import Mymodel
from hmt.optim import Optim
from hmt.trainer import Trainer
import torch
import multiprocessing as mp
import torch.distributed as dist
from hmt.data import DataDistributor
import os
from torch.multiprocessing import Queue
from hmt.data import iterator
from hmt.multiprocessing_pdb import pdb
import argparse
import options
from hmt.data import Dataset


def single_process(data_queue, rank, opt):
    if rank == 0:
        pdb.set_trace()
    if opt.num_process >1:
        for _ in range(100):
            data = data_queue.get()
            trainer.train(data)
    else:
        dataset = data_manager(None, opt)
        test_data = dataset.get_test()
        model = Mymodel(dataset.src_vocab, opt, dataset.tgt_size)
        optim = Optim(model)
        trainer = Trainer(model, optim, opt)
        for epoch in range(opt.epochs):
            itr = iterator.Iterator(dataset, opt)
            itr = itr.get_batch_iterator()
            for bn, batch in enumerate(itr):
                trainer.train(batch, test_data, epoch)
                # print(batch)


def init_process(fn, data_queue, rank):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '24580'
    dist.init_process_group(backend='gloo', rank = rank, world_size = 2)
    fn(data_queue, rank)


def data_manager(data_distributor, opt):
    if opt.num_process>1:
        dataset = Dataset()
        itr = iterator.Iterator(dataset)
        itr = itr.get_batch_iterator()
        for batch in itr:
            data_distributor.send_example_list(batch)
    else:
        dataset = Dataset(shuffle=True)
        dataset.load_from_file(opt.src_train_path, opt.tgt_train_path)
        dataset.load_test_file(opt.src_test_path, opt.tgt_test_path)
        dataset.build_vocab()
        return dataset


def main(opt):
    # mp.set_start_method('spawn')
    processes = []
    # error_queue = mp.SimpleQueue()
    data_distributor = DataDistributor()
    for rank in range(opt.num_process):
        data_queue = Queue()
        data_distributor.add(data_queue)
        p = mp.Process(target = init_process, args=(single_process, data_queue, rank))
        p.start()
        processes.append(p)
    for i in range(15):
        data_manager(data_distributor, opt)
    for p in processes:
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    options.add_train_args(parser)
    opt = parser.parse_args()
    if opt.num_process > 1:
        main(opt)
    else:
        single_process(None, None, opt)

