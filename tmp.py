import time
import numpy as np
import math
import multiprocessing as mp
from multiprocessing import shared_memory

# Replace these imports with your actual module / functions
from test import cython_loop, cyMatrixMultiply     # from your compiled Cython
def pyMatrixMultiply(a, b):
    # Python/NumPy version
    return np.matmul(a, b)

###############################################################################
# Scenario 1: Pure Python (NumPy) single-process
###############################################################################
def scenario_python_single_process(matA, vecA, num_calls=50000):
    """
    Use Python/NumPy (pyMatrixMultiply) in a single-process loop.
    """
    t1 = time.time()
    for _ in range(num_calls):
        pyMatrixMultiply(matA, vecA)
    t2 = time.time()
    return t2 - t1


###############################################################################
# Scenario 2: Cython single-process
###############################################################################
def scenario_cython_single_process(matA, vecA, num_calls=50000):
    """
    Use your compiled cyMatrixMultiply in a single-process loop.
    """
    t1 = time.time()
    for _ in range(num_calls):
        cyMatrixMultiply(matA, vecA)
    t2 = time.time()
    return t2 - t1


###############################################################################
# Scenario 3: Cython multi-process WITHOUT shared memory
#    (each worker receives a copy of the data, which can be expensive for large arrays)
###############################################################################
def worker_task_no_shared(args):
    """
    Receives matA, vecA with each call (no shared memory).
    Calls cyMatrixMultiply on its local copy.
    """
    matA_local, vecA_local = args
    # We won't return the entire result to keep overhead smaller,
    # but you could return it if needed.
    _ = cyMatrixMultiply(matA_local, vecA_local)
    return 0  # dummy return

def scenario_cython_multiprocess_no_shared(matA, vecA, num_calls=50000):
    """
    Spawns a pool; each task sends a copy of matA/vecA to the worker.
    This can be slow for large data, because each map() call serializes + sends data.
    """
    t1 = time.time()
    # We'll create a list of tuples to send to each task
    data = [(matA, vecA) for _ in range(num_calls)]

    # Use a Pool with as many processes as CPU cores
    with mp.Pool(processes=mp.cpu_count()) as pool:
        _ = pool.map(worker_task_no_shared, data)

    t2 = time.time()
    return t2 - t1


###############################################################################
# Scenario 4: Cython multi-process WITH shared memory
###############################################################################
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

# Globals (visible in each worker)
matA_shm = None
matA_glob = None
vecA_shm = None
vecA_glob = None

def init_worker_shared(mat_shm_name, mat_shape, mat_dtype_name,
                       vec_shm_name, vec_shape, vec_dtype_name):
    """
    Called once per worker when the Pool starts.
    Attaches to the shared memory blocks by name and
    creates global arrays in each worker.
    """
    global matA_shm, matA_glob, vecA_shm, vecA_glob
    matA_shm = shared_memory.SharedMemory(name=mat_shm_name)
    vecA_shm = shared_memory.SharedMemory(name=vec_shm_name)

    mat_dtype = np.dtype(mat_dtype_name)
    vec_dtype = np.dtype(vec_dtype_name)

    matA_glob = np.ndarray(mat_shape, dtype=mat_dtype, buffer=matA_shm.buf)
    vecA_glob = np.ndarray(vec_shape, dtype=vec_dtype, buffer=vecA_shm.buf)

def worker_task_shared(_):
    """
    Each task uses the globally shared matA_glob, vecA_glob.
    """
    _ = cyMatrixMultiply(matA_glob, vecA_glob)
    return 0  # dummy return

def scenario_cython_multiprocess_shared(matA, vecA, num_calls=50000):
    """
    Spawns a pool that attaches to the same shared memory blocks
    for matA, vecA, and calls cyMatrixMultiply in parallel.
    """
    # 1) Create shared memory blocks
    mat_shm, shared_matA = create_shared_array(matA)
    vec_shm, shared_vecA = create_shared_array(vecA)

    # 2) Build init_worker arguments
    initargs = (
        mat_shm.name, shared_matA.shape, shared_matA.dtype.name,
        vec_shm.name, shared_vecA.shape, shared_vecA.dtype.name,
    )

    t1 = time.time()
    with mp.Pool(
        processes=mp.cpu_count(),
        initializer=init_worker_shared,
        initargs=initargs
    ) as pool:
        # We just need to call worker_task_shared many times
        _ = pool.map(worker_task_shared, range(num_calls))
    t2 = time.time()

    # 3) Free the shared memory blocks in the main process
    mat_shm.close()
    mat_shm.unlink()
    vec_shm.close()
    vec_shm.unlink()

    return t2 - t1


###############################################################################
# MAIN: run all experiments
###############################################################################
if __name__ == "__main__":
    # Adjust matrix size + number of calls as desired
    N = 1000
    M = 1000
    num_calls = 20000

    # Create data
    matA = np.zeros((N, M), dtype=np.uint8)
    vecA = np.zeros(M,      dtype=np.uint8)

    ############################################################################
    # 1) Pure Python (NumPy) single-process
    ############################################################################
    time_sp_python = scenario_python_single_process(matA, vecA, num_calls)
    print(f"[Scenario 1: Python single-process] Time = {time_sp_python:.4f} sec")

    ############################################################################
    # 2) Cython single-process
    ############################################################################
    time_sp_cython = scenario_cython_single_process(matA, vecA, num_calls)
    print(f"[Scenario 2: Cython single-process] Time = {time_sp_cython:.4f} sec")

    ############################################################################
    # 3) Cython multi-process WITHOUT shared memory
    ############################################################################
    time_mp_no_shared = scenario_cython_multiprocess_no_shared(matA, vecA, num_calls)
    print(f"[Scenario 3: Cython multiprocess - NO shared mem] Time = {time_mp_no_shared:.4f} sec")

    ############################################################################
    # 4) Cython multi-process WITH shared memory
    ############################################################################
    time_mp_shared = scenario_cython_multiprocess_shared(matA, vecA, num_calls)
    print(f"[Scenario 4: Cython multiprocess - WITH shared mem] Time = {time_mp_shared:.4f} sec")

    print("\n=== Summary ===")
    print(f"Python single-process:    {time_sp_python:.4f} sec")
    print(f"Cython single-process:    {time_sp_cython:.4f} sec")
    print(f"Cython MP no-shared:      {time_mp_no_shared:.4f} sec")
    print(f"Cython MP shared:         {time_mp_shared:.4f} sec")
