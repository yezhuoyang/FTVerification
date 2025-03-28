from QEPG import *
import matplotlib.pyplot as plt



'''
Plot the distribuition of logical error rate for different number of errors
'''
def plot_distribution():
    distance=5
    circuit=CliffordCircuit(3)
    circuit.set_error_rate(0.00001)
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit) 


    sampler=WSampler(circuit)


    sampler.construct_QPEG()


    shot=500

    sampler.set_shots(shot)
    distribution=sampler.calc_logical_error_distribution(wlist=range(0,100))
    
    #print(distribution)

    # --- PLOT THE DISTRIBUTION ---
    # Assuming 'distribution' is something like {error_count: probability}
    keys = list(range(sampler._totalnoise))[:100]
    values = [distribution[k] for k in keys]

    plt.bar(keys, values)
    plt.xlabel("Number of Errors")
    plt.ylabel("Logical Error Rate")
    plt.title(f"Surface code with distance {distance}, shot={shot}")

    # Add a horizontal dashed red line at y=0.5
    plt.axhline(y=0.5, color='red', linestyle='--')

    # Save the figure to a PNG file
    plt.savefig("distribution.png", dpi=300)
    plt.show()


'''
Plot the physical error rate with respect to the logical error rate
'''
def compare_error_rate():
    distance=5
    error_rate_list = np.logspace(-8, -1, 20)
    logical_error_rate_list=[] 
    for er in error_rate_list:
        #print(er)
        circuit=CliffordCircuit(3)
        circuit.set_error_rate(er)
        stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
        stim_circuit=rewrite_stim_code(str(stim_circuit))
        circuit.set_stim_str(stim_circuit)
        circuit.compile_from_stim_circuit_str(stim_circuit) 


        sampler=WSampler(circuit)


        sampler.construct_QPEG()


        shot=500

        sampler.set_shots(shot)
        distribution=sampler.calc_logical_error_distribution(wlist=[15])

        logical_error_rate_list.append(distribution[15])
        print(distribution[15])
    # Plot physical error rate vs. logical error rate on a log-scale for x-axis
    plt.plot(error_rate_list, logical_error_rate_list, marker='o')
    plt.xscale('log')  # Set the x-axis to log scale
    plt.xlabel('Physical Error Rate')
    plt.ylabel('Logical Error Rate')
    plt.title('Physical Error Rate vs Logical Error Rate (Log Scale on X-axis)')
    plt.show()
    


def fit_curve():
    distance=3
    circuit=CliffordCircuit(3)
    circuit.set_error_rate(0.01)
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit) 


    sampler=WSampler(circuit)


    sampler.construct_QPEG()


    shot=10000

    sampler.set_shots(shot)

    wlist=[1, 4,5,7, 12,13,20,24, 25, 50, 70, 90,100]
    distribution=sampler.calc_logical_error_distribution(wlist=wlist)

    
    alpha=sampler.fit_curve(wlist)

    logical_error_rate=sampler.calc_logical_error_rate_by_curve_fitting(alpha)

    print("Logical error rate: ",logical_error_rate)



    xlist=np.linspace(1, 100, 1000)
    ylist=[model_function(x, alpha) for x in xlist]


    # Plot the fitted curve
    plt.plot(wlist, [distribution[x] for x in wlist], marker='o', label='Fitted Curve')
    plt.plot(xlist, ylist, label='Fitted Function')
    plt.xlabel('Number of Errors')
    plt.ylabel('Logical Error Rate')
    plt.title('Fitted Curve for Logical Error Rate')
    plt.legend()
    plt.show()











if __name__ == "__main__":

    #plot_distribution()
    #compare_error_rate()
    fit_curve()