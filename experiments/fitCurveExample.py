'''
Generate the ground truth of logical error rate for surface code and several benchmark
'''
from QEPG import *
import matplotlib.pyplot as plt




def count_logical_errors(circuit: stim.Circuit, num_shots: int) -> int:
    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    # Configure a decoder using the circuit.
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    # Run the decoder.
    predictions = matcher.decode_batch(detection_events)

    # Count the mistakes.
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    return num_errors






def surface_groundTruth():
    distance=3
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.0001)
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit)          


    

    new_stim_circuit=circuit.get_stim_circuit()        

    num_shots = 500
    num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)


    error_rate= num_logical_errors / num_shots
    print(f"Distance: {distance}, Qubit number: {circuit.get_qubit_num()}, Total Measure: {circuit.get_totalMeas()}, Total Noise: {circuit.get_totalnoise()}, Error rate: {error_rate:.9e}")



    distance=5
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.0001)
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit)           
    new_stim_circuit=circuit.get_stim_circuit()        

    num_shots = 5000
    num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)


    error_rate= num_logical_errors / num_shots
    print(f"Distance: {distance}, Qubit number: {circuit.get_qubit_num()}, Total Measure: {circuit.get_totalMeas()}, Total Noise: {circuit.get_totalnoise()}, Error rate: {error_rate:.9e}")


    distance=7
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.0001)
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit)           
    new_stim_circuit=circuit.get_stim_circuit()        

    num_shots = 5000
    num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)


    error_rate= num_logical_errors / num_shots
    print(f"Distance: {distance}, Qubit number: {circuit.get_qubit_num()}, Total Measure: {circuit.get_totalMeas()}, Total Noise: {circuit.get_totalnoise()}, Error rate: {error_rate:.9e}")




'''
Plot the distribuition of logical error rate for different number of errors
'''
def plot_distribution():
    distance=3
    circuit=CliffordCircuit(3)
    circuit.set_error_rate(0.001)
    stim_circuit=stim.Circuit.generated("repetition_code:memory",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit) 


    sampler=WSampler(circuit)


    sampler.construct_QPEG()


    shot=50

    sampler.set_shots(shot)
    distribution=sampler.calc_logical_error_distribution(wlist=range(0,100),sList=[500]*100)
    
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






def calc_logical_error():
    distance=3
    circuit=CliffordCircuit(3)
    circuit.set_error_rate(0.0001)
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit) 


    qubut_num=circuit.get_qubit_num()
    print("Qubit num: ",qubut_num)

    sampler=WSampler(circuit)


    sampler.construct_QPEG()

    sampler.set_shots(100000)

    logicalER,var=sampler.calc_logical_error_rate()
    print("Logical error rate: ",logicalER)
    print("Variance: ",var)







def fit_curve():
    distance=3
    circuit=CliffordCircuit(3)
    circuit.set_error_rate(0.00001)
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit) 


    qubut_num=circuit.get_qubit_num()
    print("Qubit num: ",qubut_num)

    print("Total noise: ",circuit.get_totalnoise())


    sampler=WSampler(circuit)


    sampler.construct_QPEG()


    shot=9987

    ground_truth=2.32e-07

    #lp=sampler.binary_search_zero(1,100,500)
    lp=2
    rp=13

    #rp=sampler.binary_search_half(1,100,600,epsilon=0.03)
    wlist=np.linspace(lp, rp, 12)
    #wlist=list(range(lp, rp+1, 1))

    wlist=[int(x) for x in wlist]
    #wlist=list(range(lp, rp+1, 1))

    print("Wlist:")
    print(wlist)

    wlist=[int(x) for x in wlist]

    sList=[shot//2]+[max(shot//4,1)]+[max((shot-shot//2-shot//4)//10,1)]*10
    print("Slist:")
    print(sList)
    distribution=sampler.sequential_calc_logical_error_distribution(wlist=wlist,sList=[shot//2]+[max(shot//4,1)]+[max((shot-shot//2-shot//4)//10,1)]*10)


    mu,alpha=sampler.fit_curve(wlist)

    logical_error_rate=sampler.calc_logical_error_rate_by_curve_fitting(lp,mu,alpha)

    accuracy = abs((logical_error_rate - ground_truth) / ground_truth) * 100

    print("Logical error rate: ",logical_error_rate)

    print("Accuracy: ",accuracy)

    xlist=np.linspace(1, 100, 1000)
    ylist=[model_function(x,mu, alpha) for x in xlist]


    print(distribution)


    print([distribution[x] for x in wlist])
    
    # Plot the fitted curve
    plt.plot(wlist, [distribution[x] for x in wlist], marker='o', label='Fitted Curve')
    plt.plot(xlist, ylist, label='Fitted Function')

    plt.scatter(lp, model_function(lp,mu, alpha),marker='*', color='red', label='Left Point')
    plt.scatter(rp, model_function(rp,mu, alpha),marker='*', color='green', label='Right Point')


    plt.xlabel('Number of Errors')
    plt.ylabel('Logical Error Rate')
    plt.title('Fitted Curve for Logical Error Rate')
    plt.legend()
    #plt.show()
    plt.savefig("fit_curve.png", dpi=300)








def logic_error_rate():
    distance=3
    circuit=CliffordCircuit(3)
    circuit.set_error_rate(0.001)
    stim_circuit=stim.Circuit.generated("repetition_code:memory",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit) 


    qubut_num=circuit.get_qubit_num()
    print("Qubit num: ",qubut_num)

    print("Total noise: ",circuit.get_totalnoise())


    sampler=WSampler(circuit)


    sampler.construct_QPEG()


    shot=1000
    sampler.set_shots(shot)


    lr=sampler.calc_logical_error_rate()

    print("Logical error rate: ",lr)






'''
Plot the distribuition of logical error rate for different number of errors
'''
def example_distribution():
    distance=3
    circuit=CliffordCircuit(3)
    circuit.set_error_rate(0.001)
    stim_circuit=stim.Circuit(
                                        '''
                                        CX 0 3
                                        CX 1 3
                                        CX 1 4
                                        CX 2 4
                                        M 3
                                        DETECTOR(1, 0) rec[-1]
                                        M 4
                                        DETECTOR(2, 0) rec[-1]
                                        M 0
                                        OBSERVABLE_INCLUDE(0) rec[-1]
                                        '''
    )
    stim_circuit=rewrite_stim_code(str(stim_circuit))



    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit) 
    print(circuit.get_stim_circuit())
    print(f"Total noise: {circuit.get_totalnoise()}")
    sampler=WSampler(circuit)


    sampler.construct_QPEG()


    shot=50

    sampler.set_shots(shot)
    distribution=sampler.sequential_calc_logical_error_distribution(wlist=range(0,11),sList=[500]*11)
    
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
Plot the distribuition of logical error rate for different number of errors
'''
def example_distribution_in_paper():
    distance=3
    circuit=CliffordCircuit(5)
    circuit.set_error_rate(0.001)
    stim_circuit=stim.Circuit(
                                        '''
                                        CX 0 3
                                        CX 1 3
                                        CX 1 4
                                        CX 2 4
                                        M 3
                                        DETECTOR(1, 0) rec[-1]
                                        M 4
                                        DETECTOR(2, 0) rec[-1]
                                        M 0
                                        OBSERVABLE_INCLUDE(0) rec[-1]
                                        '''
    )
    stim_circuit=rewrite_stim_code(str(stim_circuit))


    circuit.set_parityMatchGroup([[0],[1]])
    circuit.set_observable([2])
    #circuit.add_depolarize(0)
    #circuit.add_depolarize(1)
    #circuit.add_depolarize(2)
    #circuit.add_depolarize(3)
    #circuit.add_depolarize(4)
    circuit.add_xflip_noise(0)
    circuit.add_xflip_noise(1)
    circuit.add_xflip_noise(2)
    circuit.add_xflip_noise(3)
    circuit.add_xflip_noise(4)



    circuit.add_cnot_no_noise(0,3)
    circuit.add_cnot_no_noise(1,3)

    #circuit.add_depolarize(0)
    #circuit.add_depolarize(1)  
    circuit.add_xflip_noise(0)
    circuit.add_xflip_noise(1)

    
    circuit.add_cnot_no_noise(1,4)
    circuit.add_cnot_no_noise(2,4)

    circuit.add_measurement_no_noise(3)
    circuit.add_measurement_no_noise(4)

    circuit.add_measurement_no_noise(0)

    circuit.compile_detector_and_observable()


    print(circuit.get_stim_circuit())
    print(f"Total noise: {circuit.get_totalnoise()}")
    sampler=WSampler(circuit)

    sampler.construct_QPEG()


    shot=50

    sampler.set_shots(shot)
    distribution=sampler.sequential_calc_logical_error_distribution(wlist=range(0,8),sList=[1000]*8)
    
    #print(distribution)

    # --- PLOT THE DISTRIBUTION ---
    # Assuming 'distribution' is something like {error_count: probability}
    keys = list(range(sampler._totalnoise+1))[:100]
    values = [distribution[k] for k in keys]

    # Assuming `values` and `keys` are computed as in your provided code
    keys = list(range(len(values)))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(keys, values, color='skyblue')

    # Adding annotations
    for bar, value in zip(bars, values):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{value:.4f}', ha='center', va='bottom', fontsize=10)

    plt.xlabel('Number of Errors', fontsize=12)
    plt.ylabel('Logical Error Rate', fontsize=12)
    plt.title('Distribution of Logical Error Rate in subspaces with Different Number of Errors', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # Save the figure to a PNG file
    plt.savefig("distributionexample.png", dpi=300)
    plt.show()


if __name__ == "__main__":


    #fit_curve()
    #plot_distribution()
    #logic_error_rate()
    #calc_logical_error()

    example_distribution_in_paper()