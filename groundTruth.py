'''
Generate the ground truth of logical error rate for surface code and several benchmark
'''
from QEPG import *





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

    num_shots = 100
    num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)



    error_rate= num_logical_errors / num_shots
    print(f"Distance: {distance}, Qubit number: {circuit.get_qubit_num()}, Total Measure: {circuit.get_totalMeas()}, Total Noise: {circuit.get_totalnoise()}, Error rate: {error_rate:.9e}")



    distance=5
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.0005)
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit)           
    new_stim_circuit=circuit.get_stim_circuit()        

    num_shots =1000
    num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)


    error_rate= num_logical_errors / num_shots
    print(f"Distance: {distance}, Qubit number: {circuit.get_qubit_num()}, Total Measure: {circuit.get_totalMeas()}, Total Noise: {circuit.get_totalnoise()}, Error rate: {error_rate:.9e}")


    distance=7
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.0005)
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)


    circuit.compile_from_stim_circuit_str(stim_circuit)           
    new_stim_circuit=circuit.get_stim_circuit()        

    num_shots =10000000
    num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)


    error_rate= num_logical_errors / num_shots
    print(f"Distance: {distance}, Qubit number: {circuit.get_qubit_num()}, Total Measure: {circuit.get_totalMeas()}, Total Noise: {circuit.get_totalnoise()}, Error rate: {error_rate:.9e}")





def repitition_groundTruth():
    distance=3
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.001)
    stim_circuit=stim.Circuit.generated("repetition_code:memory",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit)           
    new_stim_circuit=circuit.get_stim_circuit()        

    num_shots = 50000000
    num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)


    error_rate= num_logical_errors / num_shots
    print(f"Distance: {distance}, Qubit number: {circuit.get_qubit_num()}, Total Measure: {circuit.get_totalMeas()}, Total Noise: {circuit.get_totalnoise()}, Error rate: {error_rate:.9e}")



    distance=5
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.001)
    stim_circuit=stim.Circuit.generated("repetition_code:memory",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit)           
    new_stim_circuit=circuit.get_stim_circuit()        

    num_shots = 500
    num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)


    error_rate= num_logical_errors / num_shots
    print(f"Distance: {distance}, Qubit number: {circuit.get_qubit_num()}, Total Measure: {circuit.get_totalMeas()}, Total Noise: {circuit.get_totalnoise()}, Error rate: {error_rate:.9e}")


    distance=7
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.003)
    stim_circuit=stim.Circuit.generated("repetition_code:memory",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit)           
    new_stim_circuit=circuit.get_stim_circuit()        

    num_shots = 500
    num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)


    error_rate= num_logical_errors / num_shots
    print(f"Distance: {distance}, Qubit number: {circuit.get_qubit_num()}, Total Measure: {circuit.get_totalMeas()}, Total Noise: {circuit.get_totalnoise()}, Error rate: {error_rate:.9e}")






def color_groundTruth():
    distance=3
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.0001)
    stim_circuit=stim.Circuit.generated("color_code:memory_xyz",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit)           
    new_stim_circuit=circuit.get_stim_circuit()        

    num_shots = 100
    num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)


    error_rate= num_logical_errors / num_shots
    print(f"Distance: {distance}, Qubit number: {circuit.get_qubit_num()}, Total Measure: {circuit.get_totalMeas()}, Total Noise: {circuit.get_totalnoise()}, Error rate: {error_rate:.9e}")



    distance=5
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.0001)
    stim_circuit=stim.Circuit.generated("color_code:memory_xyz",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit)           
    new_stim_circuit=circuit.get_stim_circuit()        

    num_shots = 100
    num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)


    error_rate= num_logical_errors / num_shots
    print(f"Distance: {distance}, Qubit number: {circuit.get_qubit_num()}, Total Measure: {circuit.get_totalMeas()}, Total Noise: {circuit.get_totalnoise()}, Error rate: {error_rate:.9e}")


    distance=7
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.0001)
    stim_circuit=stim.Circuit.generated("color_code:memory_xyz",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit)           
    new_stim_circuit=circuit.get_stim_circuit()        

    num_shots = 100
    num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)


    error_rate= num_logical_errors / num_shots
    print(f"Distance: {distance}, Qubit number: {circuit.get_qubit_num()}, Total Measure: {circuit.get_totalMeas()}, Total Noise: {circuit.get_totalnoise()}, Error rate: {error_rate:.9e}")




def Heavy_hexagon():
    pass  



def Heavy_Square():
    pass





def stim_samples_repitition():
    distance=3
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.001)
    stim_circuit=stim.Circuit.generated("repetition_code:memory",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit)           
    new_stim_circuit=circuit.get_stim_circuit()        

    shots_list=[100,500,800,1000,2000,3000,5000,8000,10000,20000,30000,50000,80000,100000,200000,300000,500000,800000,1000000]

    ground_truth=6.38e-5

    for shot in shots_list:

        #Repeat the experiment for 10 times, take the average of the error rate
        error_rate_list=[]
        for i in range(30):

            num_shots = shot
            num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)
            error_rate= num_logical_errors / num_shots
            error_rate_list.append(error_rate)
        
        error_rate=np.mean(error_rate_list)
        
        print(f"Shot: {shot}, Error rate: {error_rate:.9e}")

        accuracy=abs((error_rate-ground_truth)/ground_truth)*100
        print(f"Accuracy: {accuracy:.4f}")
            









if __name__ == "__main__":
    #QEPG_sample()   
    #compare_two_method()
    #stim_sample()
    #print("Surface code ground truth")
    #surface_groundTruth()
    #print("Repitition code ground truth")
    #repitition_groundTruth()



    stim_samples_repitition()
