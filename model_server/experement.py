import torch
from torch.multiprocessing import Queue, Process, Pipe
import os
import time

q = Queue()
def f(name):
    print('hello', name)
    time.sleep(2)

p = Process(name='Hubert',target=f, args=('bob',))
tn = torch.rand((2,2))
p.start()
print(os.system('ps -A'))
p.join()
