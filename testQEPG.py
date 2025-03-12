#from QEPG import *
from test import cython_loop,cyMatrixMultiply
import time
import math
import numpy as np
import multiprocessing as mp



def python_loop():
    

    sum=0
    for i in range(100000000):
        sum+=i
        sum=sum%1000
    return sum





def pyStdDev(a):
    mean = sum(a) / len(a)
    return math.sqrt((sum(((x - mean)**2 for x in a)) / len(a)))


def npStdDev(a):
    return np.std(a)

import concurrent.futures


def pyMatrixMultiply(a,b):
    return np.matmul(a,b)

from multiprocessing import shared_memory

def create_shared_array(initial_array: np.ndarray):
    """
    Create a new shared memory block of the same size as `initial_array`,
    copy the data into it, and return (shm, shm_array).
    """
    shm = shared_memory.SharedMemory(create=True, size=initial_array.nbytes)
    shm_array = np.ndarray(initial_array.shape,
                           dtype=initial_array.dtype,
                           buffer=shm.buf)
    # Copy data into shared memory
    shm_array[:] = initial_array[:]
    return shm, shm_array


# Globals (in each worker) referencing the shared arrays
matA_shm = None
matA = None
vecA_shm = None
vecA = None

def init_worker(mat_shm_name, mat_shape, mat_dtype_name,
                vec_shm_name, vec_shape, vec_dtype_name):
    """
    Called ONCE per worker process when the Pool starts.
    Attaches to the shared memory blocks by name.
    """
    global matA_shm, matA, vecA_shm, vecA
    from multiprocessing import shared_memory

    matA_shm = shared_memory.SharedMemory(name=mat_shm_name)
    vecA_shm = shared_memory.SharedMemory(name=vec_shm_name)

    mat_dtype = np.dtype(mat_dtype_name)
    vec_dtype = np.dtype(vec_dtype_name)

    matA = np.ndarray(mat_shape, dtype=mat_dtype, buffer=matA_shm.buf)
    vecA = np.ndarray(vec_shape, dtype=vec_dtype, buffer=vecA_shm.buf)

def worker_task(_):
    """
    Each pool task uses the globally shared matA, vecA.
    """
    # Perform the matrix-vector multiply with the shared arrays
    return cyMatrixMultiply(matA, vecA)
    # If the result is large, note that returning it can also be big overhead.
    # You might store results back into shared memory or do something else.

if __name__ == "__main__":


    time1=time.time()
    N=1000
    M=1000
    matA=np.zeros((N,M),dtype=np.uint8)
    vecA=np.zeros(M,dtype=np.uint8)

    for i in range(50000):
        pyMatrixMultiply(matA,vecA)
    time2=time.time()
    print("Time taken: ",time2-time1)



    time1=time.time()
    N=1000
    M=1000
    matA=np.zeros((N,M),dtype=np.uint8)
    vecA=np.zeros(M,dtype=np.uint8)

    for i in range(50000):
        cyMatrixMultiply(matA,vecA)
    time2=time.time()
    print("Time taken: ",time2-time1)



    time1=time.time()
    N=1000
    M=1000
    matA=np.zeros((N,M),dtype=np.uint8)
    vecA=np.zeros(M,dtype=np.uint8)
    # 2) Create shared memory blocks
    mat_shm, shared_matA = create_shared_array(matA)
    vec_shm, shared_vecA = create_shared_array(vecA)

    # 3) Build init_worker arguments
    initargs = (
        mat_shm.name, shared_matA.shape, shared_matA.dtype.name,
        vec_shm.name, shared_vecA.shape, shared_vecA.dtype.name,
    )

    # 4) Spawn a Pool that calls init_worker once per process
    num_calls = 50000
    t1 = time.time()
    with mp.Pool(
        processes=mp.cpu_count(),
        initializer=init_worker,
        initargs=initargs
    ) as pool:
        # Map 2000 tasks to the pool. Each worker sees the same shared memory.
        _results = pool.map(worker_task, range(num_calls))

    t2 = time.time()

    # 5) Free the shared memory blocks
    mat_shm.close()
    mat_shm.unlink()
    vec_shm.close()
    vec_shm.unlink()

    print("Time taken: ",t2-t1)


    #result=cyMatrixMultiply(matA,vecA)
    #print(result)

    '''
    print("Cython loop:")
    time1=time.time()
    test=cython_loop()
    time2=time.time()
    print(test)
    print("Time taken: ",time2-time1)


    print("Python loop:")
    time1=time.time()
    test2=python_loop()
    time2=time.time()
    print(test2)
    print("Time taken: ",time2-time1)
    '''
