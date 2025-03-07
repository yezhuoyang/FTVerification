# noise_sampler.pyx
#import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free, rand, RAND_MAX
from multiprocessing import shared_memory
from libc.time cimport time_t, time
import random  # Import Python's random module
# Seed the random number generator once at module initialization
cdef extern from "stdlib.h":
    void srand(unsigned int seed)

# Initialize the seed when the module is imported


'''
@cython.boundscheck(False)
@cython.wraparound(False)
def sample_fixed_one_two_three(int N, int k):
    """
    Returns a list of length N containing exactly k non-zero elements (1, 2, or 3),
    each chosen from 1, 2, 3 with equal probability, in random order,
    and N-k zeros.
    """
    # Seed the random number generator with current time
    srand(<unsigned int>time(NULL))
    
    # Allocate arrays for arr (ones and zeros) and arrtype (1, 2, 3)
    cdef int* arr = <int*>malloc(N * sizeof(int))
    cdef int* arrtype = <int*>malloc(N * sizeof(int))
    if arr == NULL or arrtype == NULL:
        if arr != NULL: free(arr)
        if arrtype != NULL: free(arrtype)
        raise MemoryError("Failed to allocate memory")
    
    cdef int i, j, temp
    
    # Step 1: Initialize arr with k ones and N-k zeros
    for i in range(k):
        arr[i] = 1
    for i in range(k, N):
        arr[i] = 0
    
    # Step 2: Initialize arrtype with random 1, 2, or 3
    for i in range(N):
        arrtype[i] = rand() % 3 + 1  # 1, 2, or 3 with equal probability
    
    # Step 3: Shuffle both arrays using Fisher-Yates
    # Shuffle arr
    for i in range(N - 1, 0, -1):
        j = rand() % (i + 1)
        temp = arr[i]
        arr[i] = arr[j]
        arr[j] = temp
    
    # Shuffle arrtype
    for i in range(N - 1, 0, -1):
        j = rand() % (i + 1)
        temp = arrtype[i]
        arrtype[i] = arrtype[j]
        arrtype[j] = temp
    
    # Step 4: Combine element-wise (arr * arrtype)
    cdef list py_result = [arr[i] * arrtype[i] for i in range(N)]
    
    # Free memory
    free(arr)
    free(arrtype)
    
    return py_result
'''


def sample_fixed_one_two_three(N, k):
    """
    Returns a list of length N containing exactly k ones 
    (and N-k zeros), in a random order.
    """
    # Step 1: Create a list of k ones and N-k zeros
    arr = [1]*k + [0]*(N-k)
    
    # Step 2: Create a list of 1 or two
    arrtype=[]
    
    for i in range(N):
        arrtype.append(random.randint(1,3))

    
    # Step 2: Shuffle the list randomly
    random.shuffle(arr)
    random.shuffle(arrtype)
    
    return [a * b for a, b in zip(arr, arrtype)]



# Assume sample_fixed_one_two_three is provided elsewhere; we'll call it as-is
# If it’s Python, we’ll keep it, but ideally, it’d be Cythonized too.

@cython.profile(True)
@cython.boundscheck(False)  # Disable bounds checking for speed
@cython.wraparound(False)   # Disable negative indexing for speed
def sample_noise_and_calc_result(
    int shots, 
    int totalnoise, 
    int total_meas, 
    W,  # Weight parameter, type depends on sample_fixed_one_two_three
    str dtype, 
    str shm_detec_name, 
    list parity_group, 
    list observable
):
    # Motivation: Declare all variables with C types for speed
    cdef int i, j, parity, random_val
    cdef int three_totalnoise = 3 * totalnoise  # Precompute for efficiency
    
    # Access shared memory and create NumPy array
    shm_detec = shared_memory.SharedMemory(name=shm_detec_name)
    cdef np.ndarray[np.uint8_t, ndim=2] detectorMatrix = np.ndarray(
        (total_meas, three_totalnoise), dtype=dtype, buffer=shm_detec.buf
    )
    
    # Motivation: Use memoryviews for fast array access
    cdef unsigned char[:] noise_vector = np.zeros(three_totalnoise, dtype=np.uint8)
    cdef unsigned char[:] detector_result
    
    # Motivation: Pre-allocate result lists as NumPy arrays for speed
    cdef np.ndarray[np.uint8_t, ndim=2] detection_events = np.zeros(
        (shots, len(parity_group)), dtype=np.uint8
    )
    cdef np.ndarray[np.uint8_t, ndim=2] observable_flips = np.zeros(
        (shots, 1), dtype=np.uint8  # Assuming observable flip is a single value per shot
    )
    

    # Main loop over shots
    for i in range(shots):
        # Motivation: Call external sampling function (assumed Python for now)
        random_index = sample_fixed_one_two_three(totalnoise, W)
        
        # Reset noise vector efficiently
        for j in range(three_totalnoise):
            noise_vector[j] = 0
        
        # Motivation: Use C-style loop for noise vector construction
        for j in range(totalnoise):
            random_val = random_index[j]
            if random_val == 1:
                noise_vector[j] = 1
            elif random_val == 2:
                noise_vector[j + totalnoise] = 1
            elif random_val == 3:
                noise_vector[j + 2 * totalnoise] = 1
        
        # Motivation: Efficient matrix multiplication with NumPy, then use memoryview
        detector_result = np.matmul(detectorMatrix, noise_vector) % 2
        
        # Compute detection events for parity groups
        for group_idx, group in enumerate(parity_group):
            parity = 0
            for idx in group:
                if detector_result[idx] == 1:
                    parity += 1
            detection_events[i, group_idx] = parity % 2
        
        # Compute observable flips
        parity = 0
        for idx in observable:
            if detector_result[idx] == 1:
                parity += 1
        observable_flips[i, 0] = parity % 2
    
    # Convert to Python-compatible lists if needed (optional)
    return detection_events.tolist(), observable_flips.tolist()





# We use "int64_t" here; adapt as needed
ctypedef cnp.int64_t DTYPE_t

def cython_sample_noise_and_calc_result(
    int shots, 
    int totalnoise, 
    object W,              # Because W might be Python-specific; adapt if numeric
    object dtype,          # The NumPy dtype object
    str shm_detec_name, 
    int parity_group_length
):
    """
    Cython version of 'python_sample_noise_and_calc_result'.
    """
    # --- cdef declarations at the top ---
    cdef object shm_detec  # Must be 'object', not a cimport from shared_memory
    cdef np.ndarray detectorMatrix_np
    cdef np.int64_t[:, :] detectorMatrix

    cdef list detection_events = []
    cdef list observable_flips = []

    cdef int shot, i, row, col
    cdef long s

    cdef np.ndarray random_index_np
    cdef np.int64_t[::1] random_index

    cdef np.ndarray noise_vector_np
    cdef np.int64_t[::1] noise_vector

    cdef np.ndarray detectorresult_np
    cdef np.int64_t[::1] detectorresult
    # --- end cdef declarations ---

    # 1) Attach to existing shared memory and construct a NumPy array from it
    shm_detec = shared_memory.SharedMemory(name=shm_detec_name)
    detectorMatrix_np = np.ndarray(
        shape=(parity_group_length + 1, 3 * totalnoise),
        dtype=dtype,
        buffer=shm_detec.buf
    )

    # 2) Convert that NumPy array to a typed Cython memoryview
    detectorMatrix = detectorMatrix_np

    # 3) Main loop over shots
    for shot in range(shots):
        # 3.1) Generate random indices
        random_index_np = sample_fixed_one_two_three(totalnoise, W)
        random_index = random_index_np

        # 3.2) Build the noise_vector (length 3*totalnoise)
        noise_vector_np = np.zeros(3 * totalnoise, dtype=np.int64)
        noise_vector = noise_vector_np

        for i in range(totalnoise):
            if random_index[i] == 1:
                noise_vector[i] = 1
            elif random_index[i] == 2:
                noise_vector[i + totalnoise] = 1
            elif random_index[i] == 3:
                noise_vector[i + 2 * totalnoise] = 1

        # 3.3) Multiply detectorMatrix * noise_vector modulo 2
        detectorresult_np = np.zeros(parity_group_length + 1, dtype=np.int64)
        detectorresult = detectorresult_np

        for row in range(parity_group_length + 1):
            s = 0
            for col in range(3 * totalnoise):
                s += detectorMatrix[row, col] * noise_vector[col]
            detectorresult[row] = s & 1  # same as s % 2, but faster

        # 3.4) Split out the first parity_group_length bits vs. last bit
        detection_events.append(detectorresult_np[:parity_group_length].tolist())
        observable_flips.append(detectorresult_np[parity_group_length:].tolist())

    return detection_events, observable_flips