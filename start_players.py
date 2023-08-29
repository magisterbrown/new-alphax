from multiprocessing import Pool
from play import play
import time

def start_play(idx: int):
    while True:
        play() 

    return idx 

if __name__ == '__main__':
    processes = 9
    with Pool(processes) as p:
        res = p.map(start_play, range(processes))
        print(res)
