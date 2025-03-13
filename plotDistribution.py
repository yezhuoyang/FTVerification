from QEPG import *
import matplotlib.pyplot as plt



'''
Plot the distribuition of logical error rate for different number of errors
'''
def plot_distribution():
    distance=5
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.001)
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit) 


    sampler=WSampler(circuit)


    sampler.construct_QPEG()


    shot=1000

    sampler.set_shots(shot)
    distribution=sampler.calc_logical_error_distribution()
    
    print(distribution)

    # --- PLOT THE DISTRIBUTION ---
    # Assuming 'distribution' is something like {error_count: probability}
    keys = list(range(sampler._totalnoise))[:40]
    values = [distribution[k] for k in keys]

    plt.bar(keys, values)
    plt.xlabel("Number of Errors")
    plt.ylabel("Logical Error Rate")
    plt.title("Distribution of Logical Error Rate")

    # Save the figure to a PNG file
    plt.savefig("distribution.png", dpi=300)
    plt.show()

if __name__ == "__main__":

    plot_distribution()