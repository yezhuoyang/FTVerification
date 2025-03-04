#Compare the running time of Stim and my method
from QEPG import *
from typing import List
import sinter
import matplotlib.pyplot as plt
import os


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

       
        self._sampler.set_shots(10000)
        self._sampler.construct_detector_model()


    def calc_threhold(self):
        logical_list=[]
        dvals=[3,5,7]
        noise_list=[0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0045, 0.0050, 0.01, 0.015, 0.020, 0.025, 0.030]
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


if __name__ == "__main__":

    surf=mySurface()
    surf.calc_threhold()
