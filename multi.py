

from multiprocessing import Pool
import time


def f(x):
    return x*x


import random

def one_sample(length):
    strtmp=""
    for i in range(length):
        strtmp+=str(random.randint(0,1))
    return strtmp


#Parallel sampling
'''
Generate N samples of 01 string in parallel(N different process)
'''
def parallel_sample(N,length):
    with Pool(processes=N) as pool:
        results=pool.map(one_sample,[length]*N)
    return results


def non_parallel_sample(N,length):
    strtmp=""
    for i in range(N*length):
        strtmp+=str(random.randint(0,1))
    return strtmp    





class Sampler:


    def __init__(self):
        self._count=0


    def one_sample(self,length):
        strtmp=""
        for i in range(length):
            strtmp+=str(random.randint(0,1))
        self._count+=1
        return strtmp
    

    def parallel_sample(self,N,length):
        with Pool(processes=N) as pool:
            results=pool.map(self.one_sample,[length]*N)
        return results








if __name__ == '__main__':
    '''
    start_time = time.time()
    samples=parallel_sample(N=60,length=500000)
    end_time = time.time()
    parallel_elapsed = end_time - start_time
    print(f"[Parallel] Time elapsed = {parallel_elapsed:.4f} seconds")

    #for s in samples:
    #    print(s)

    start_time = time.time()    
    newsamples=non_parallel_sample(N=60,length=500000)
    end_time = time.time()
    non_parallel_elapsed = end_time - start_time
    print(f"[Non-Parallel] Time elapsed = {non_parallel_elapsed:.4f} seconds")
    '''

    sampler=Sampler()
    result=sampler.parallel_sample(60,500000)
    print(sampler._count)


    print(len(result))