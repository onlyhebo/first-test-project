import multiprocessing
import os
#import pdb

from multiprocessing_pdb import pdb


def child_process():
    print('child process')
    pdb.set_trace()
    print('hello pdb')
'''
    p_in = os.open('p_in', os.O_RDWR|os.O_NOCTTY)
    p_out = os.open('p_out', os.O_RDWR|os.O_NOCTTY)
    pdb.Pdb(stdin=open(p_in, 'r+'), stdout=open(p_out, 'w+')).set_trace()'''

def main_process():
    print('parent-process')
    processes = []
    for i in range(2):
        p = multiprocessing.Process(target=child_process)
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


if __name__ == '__main__':
    main_process()
