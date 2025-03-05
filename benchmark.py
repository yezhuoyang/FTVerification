#Compare the running time of Stim and my method
from QEPG import *
from typing import List
import sinter
import matplotlib.pyplot as plt
import os
import cProfile
import pstats

class StimSurface():
    def __init__(self):
        pass

    def generated(self,distance,errorrate):
        stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
        stim_str=rewrite_stim_code(str(stim_circuit))
        circuit=CliffordCircuit(2)
        circuit.set_error_rate(errorrate)
        circuit.compile_from_stim_circuit_str(stim_str)

        return circuit._stimcircuit


    def calc_threhold(self):
        tasks = [
            sinter.Task(
                circuit=self.generated(
                    distance=d,
                    errorrate=noise,
                ),
                json_metadata={'d': d, 'p': noise},
            )
            for d in [3,5,7]
            for noise in [0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0045, 0.0050, 0.01, 0.015, 0.020, 0.025, 0.030]
        ]

        collected_stats: List[sinter.TaskStats] = sinter.collect(
            num_workers=os.cpu_count(),
            tasks=tasks,
            decoders=['pymatching'],
            max_shots=1_000_000,
            max_errors=5_000,
            print_progress=True
        )


        fig, ax = plt.subplots(1, 1)
        sinter.plot_error_rate(
            ax=ax,
            stats=collected_stats,
            x_func=lambda stats: stats.json_metadata['p'],
            group_func=lambda stats: stats.json_metadata['d'],
        )
        #ax.set_ylim(5e-1, 5e-2)
        #ax.set_xlim(0.000, 0.004)
        ax.loglog()
        ax.set_title("Surface Code Error Rates (Phenomenological Noise)")
        ax.set_xlabel("Phyical Error Rate")
        ax.set_ylabel("Logical Error Rate per Shot")
        ax.grid(which='major')
        ax.grid(which='minor')
        ax.legend()
        fig.set_dpi(120)  # Show it bigger
        fig.savefig("tmp.png")



class mySurface():

    def __init__(self):
        self._circuit=None
        self._tracer=None
        self._sampler=None
        self._QPEGGraph=None
        self._final_list=[]
        pass

    

    def generated(self,distance,errorrate):
        stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
        stim_str=rewrite_stim_code(str(stim_circuit))
        circuit=CliffordCircuit(2)
        circuit.set_error_rate(errorrate)
        circuit.compile_from_stim_circuit_str(stim_str)
        self._circuit=circuit

        self._sampler=WSampler(self._circuit,distance=distance)

       
        self._sampler.set_shots(200)
        self._sampler.construct_detector_model()


    def calc_threhold(self):
        logical_list=[]
        #dvals=[3,5,7]
        #noise_list=[0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0045, 0.0050, 0.01, 0.015, 0.020, 0.025, 0.030]
        dvals=[3]
        noise_list=[0.0005]
        for d in dvals:
            tmp_list=[]
            hasQP=False
            for noise in noise_list:
                self.generated(d,noise)
                if not hasQP:
                    self._sampler.construct_QPEG()
                    self._QPEG=self._sampler._QPEGraph
                    hasQP=True

                self._sampler._QPEGraph=self._QPEG
                print(f"Start sampling!  d= {d} Physical Error rate={noise }")
                self._sampler.calc_logical_error_rate()
                print(f"d= {d} Physical Error rate={noise }, Logical Error rate:{self._sampler._logical_error_rate}")
                tmp_list.append(self._sampler._logical_error_rate)
            logical_list.append(tmp_list)
        self._final_list=logical_list


        # Now make the log–log plot and save to 'tmp.png'.
        plt.figure(figsize=(6,4))
        for i, d in enumerate(dvals):
            plt.plot(noise_list, logical_list[i], marker='o', label=f"d = {d}")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Physical Error Rate")
        plt.ylabel("Logical Error Rate per Shot")
        plt.title("Repetition Code Error Rates (Phenomenological Noise)")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("tmp2.png", dpi=300)
        plt.close()





def compare_shots():

    shots=[10,20,50,100,150,200,250,500,750,1000,1500,2000,3000,4000,5000,6000,7000,10000,100000,200000,1000000]
    logical_list=[]
    niave_logical_list=[]

    for shot in shots:

        
        stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=2,distance=3).flattened()
        stim_str=rewrite_stim_code(str(stim_circuit))
        circuit=CliffordCircuit(2)
        circuit.set_error_rate(0.0001)
        circuit.compile_from_stim_circuit_str(stim_str)

        sampler=WSampler(circuit,distance=5)

        sampler.construct_QPEG()

        sampler.set_shots(shot)
        sampler.construct_detector_model()

        sampler.calc_logical_error_rate()

        print(sampler._logical_error_rate)
        logical_list.append(sampler._logical_error_rate)

        Nsampler=NaiveSampler(circuit)
        Nsampler.set_shots(shot)
        Nsampler.calc_logical_error_rate()

        print(Nsampler._logical_error_rate)
        niave_logical_list.append(Nsampler._logical_error_rate)


        # Now make the log–log plot and save to 'tmp.png'.
    plt.figure(figsize=(6,4))
    plt.plot(shots, logical_list, marker='o', label=f"QEPG")
    plt.plot(shots, niave_logical_list, marker='o', label=f"Naive")
    plt.savefig("tmp3.png", dpi=300)
    plt.close()


import time



def python_sample_noise_and_calc_result(randomindex,shots,totalnoise,total_meas,W,dtype,shm_detec_name,parity_group, observable):
    shm_detec = shared_memory.SharedMemory(name=shm_detec_name)
    dectectorMatrix = np.ndarray((total_meas,3*totalnoise), dtype=dtype, buffer=shm_detec.buf)  
    detection_events=[]
    observable_flips=[]
    for i in range(shots):
        random_index=randomindex[i]
        noise_vector=np.array([0]*3*totalnoise)
        for i in range(totalnoise):
            if random_index[i]==1:
                noise_vector[i]=1
            elif random_index[i]==2:
                noise_vector[i+totalnoise]=1
            elif random_index[i]==3:
                noise_vector[i+2*totalnoise]=1           

        detectorresult=np.matmul(dectectorMatrix, noise_vector)%2

        tmp_detection_events=[]
        for group in parity_group:
            parity=0
            for i in group:
                if detectorresult[i]==1:
                    parity+=1
            parity=parity%2
            if parity==1:
                tmp_detection_events.append(True)
            else:
                tmp_detection_events.append(False)
                
        tmp_observable_flips=[]
        parity=0
        for index in observable:
            if detectorresult[index]==1:
                parity+=1
        parity=parity%2
        if parity==1:
            tmp_observable_flips.append(1)
        else:
            tmp_observable_flips.append(0)
        detection_events.append(tmp_detection_events)
        observable_flips.append(tmp_observable_flips)
    return detection_events, observable_flips



def test_sample_noise_differences(shots=10, totalnoise=5, total_meas=4, W=2):
    """
    Test Python and Cython versions of sample_noise_and_calc_result with fixed random_index samples.
    
    Parameters:
    - shots: Number of shots to simulate
    - totalnoise: Total noise parameter
    - total_meas: Total measurements
    - W: Number of ones in random_index
    """
    # Pre-generate fixed random_index samples
    print(f"Generating {shots} fixed random_index samples...")
    random.seed(42)  # Set seed for reproducibility
    random_indices = [sample_fixed_one_two_three(totalnoise, W) for _ in range(shots)]

    # Setup shared memory with dummy data
    detectorMatrix = np.random.randint(0, 2, (total_meas, 3 * totalnoise), dtype='uint8')
    shm = shared_memory.SharedMemory(create=True, size=detectorMatrix.nbytes)
    shm_array = np.ndarray(detectorMatrix.shape, dtype='uint8', buffer=shm.buf)
    shm_array[:] = detectorMatrix[:]
    
    # Input parameters
    dtype = 'uint8'
    shm_name = shm.name
    parity_group = [[0, 1], [2, 3]]  # Example parity groups
    observable = [0, 2]              # Example observable indices

    # Run Python version
    print("Running Python version...")
    start_time = time.time()
    py_detection_events, py_observable_flips = python_sample_noise_and_calc_result(
        random_indices, shots, totalnoise, total_meas, W, dtype, shm_name, parity_group, observable
    )
    py_time = time.time() - start_time
    print(f"Python version took {py_time:.4f} seconds")

    # Run Cython version
    print("Running Cython version...")
    start_time = time.time()
    cy_detection_events, cy_observable_flips = cython_sample_noise_and_calc_result(
        random_indices, shots, totalnoise, total_meas, W, dtype, shm_name, parity_group, observable
    )
    cy_time = time.time() - start_time
    print(f"Cython version took {cy_time:.4f} seconds")

    print(cy_observable_flips)

    print(py_observable_flips)

    # Compare outputs
    print("\nComparing outputs...")
    # Convert Python’s True/False to 1/0 for comparison
    py_detection_events = [[1 if x else 0 for x in row] for row in py_detection_events]
    # Python returns lists of lists with single elements, Cython returns flat lists
    py_observable_flips = [x[0] for x in py_observable_flips]

    detection_equal = np.array_equal(py_detection_events, cy_detection_events)
    observable_equal = np.array_equal(py_observable_flips, cy_observable_flips)
    
    print(f"Detection events match: {detection_equal}")
    print(f"Observable flips match: {observable_equal}")
    
    if not detection_equal:
        print("Python detection events:", py_detection_events)
        print("Cython detection events:", cy_detection_events)
    if not observable_equal:
        print("Python observable flips:", py_observable_flips)
        print("Cython observable flips:", cy_observable_flips)

    # Performance comparison
    print(f"\nPerformance comparison:")
    print(f"Python time: {py_time:.4f} s, Cython time: {cy_time:.4f} s")
    if py_time > 0:
        print(f"Cython speedup: {py_time / cy_time:.2f}x")

    # Cleanup
    shm.close()
    shm.unlink()


if __name__ == "__main__":


    profiler = cProfile.Profile()
    profiler.enable()


    surf=mySurface()
    surf.calc_threhold()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime')  # Sort by cumulative time
    stats.print_stats(10)        # Limit to top 10 entries
    #profiler.print_stats(sort='cumtime')  # Sort by cumulative time
    #result=sample_fixed_one_two_three(10,6)
    #print(result)

    #test_sample_noise_differences(shots=2, totalnoise=50, total_meas=5, W=10)


    #result=python_sample_fixed_one_two_three(10,6)
    #print(result)    

    #result=sample_fixed_one_two_three(10,6)
    #print(result)    