# noise_sampler.pyx
import numpy as np
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





# Assume sample_fixed_one_two_three is provided elsewhere; we'll call it as-is
# If it’s Python, we’ll keep it, but ideally, it’d be Cythonized too.

@cython.boundscheck(False)  # Disable bounds checking for speed
@cython.wraparound(False)   # Disable negative indexing for speed
def cython_sample_noise_and_calc_result(
    list randomindex,
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

    # Change observable_flips to a 1D array instead of 2D
    cdef np.ndarray[np.uint8_t, ndim=2] observable_flips = np.zeros(
        (shots, 1), dtype=np.uint8  # Assuming observable flip is a single value per shot
    )

    
    # Main loop over shots
    for i in range(shots):
        # Motivation: Call external sampling function (assumed Python for now)
        random_index = randomindex[i]
        
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