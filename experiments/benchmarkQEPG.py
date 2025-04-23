from QEPG import *
from time import time
import matplotlib.pyplot as plt

# Placeholder functions to simulate processing times
def simulate_stim_processing(distance):
    circuit=CliffordCircuit(2)
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
    #stim_circuit=rewrite_stim_code(str(stim_circuit))
    #circuit.set_stim_str(stim_circuit)
    #circuit.compile_from_stim_circuit_str(stim_circuit)
    stimcircuit=stim_circuit
    time1=time()
    detector_model=stimcircuit.detector_error_model(decompose_errors=True)
    time2=time()
    T=time2-time1
    return T

def simulate_QEPG_processing(distance):

    circuit=CliffordCircuit(2)
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit)    
    QEPGgraph=QEPG(circuit)

    time1=time()
    QEPGgraph.backword_graph_construction()  
    time2=time()
    T=time2-time1
    return T


def compare_QEPG_speed():
    distances = np.arange(3,10, 1)
    stim_times = [simulate_stim_processing(d) for d in distances]
    QEPG_times = [simulate_QEPG_processing(d) for d in distances]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Stim Processing Time
    axes[0].plot(distances, stim_times, marker='o', label="Stim Processing Time")
    axes[0].set_xlabel("Code Distance")
    axes[0].set_ylabel("Processing Time (s)")
    axes[0].set_title("Stim Processing Time")
    axes[0].legend()
    axes[0].grid()

    # Plot QEPG Processing Time
    axes[1].plot(distances, QEPG_times, marker='s', label="QEPG Processing Time")
    axes[1].set_xlabel("Code Distance")
    axes[1].set_ylabel("Processing Time (s)")
    axes[1].set_title("QEPG Processing Time")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.savefig("comparison_QEPG_speed_side_by_side.png")
    plt.show()

    

if __name__ == "__main__":

    compare_QEPG_speed()