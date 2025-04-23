'''
Generate the ground truth of logical error rate for surface code and several benchmark
'''
from QEPG import *
from heavy_square_4_z_bm import heavy_square_4_circ
from hexagon_z import hexagon_3_circ



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


    distance=13
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.001)
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)


    circuit.compile_from_stim_circuit_str(stim_circuit)           
    new_stim_circuit=circuit.get_stim_circuit()        

    num_shots =5000000
    num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)


    error_rate= num_logical_errors / num_shots
    print(f"Distance: {distance}, Qubit number: {circuit.get_qubit_num()}, Total Measure: {circuit.get_totalMeas()}, Total Noise: {circuit.get_totalnoise()}, Error rate: {error_rate:.9e}")




def surface_groundTruth2():
    distance=3


    #plist=[0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001]
    #samplelist=[100000,500000,1000000,10000000,100000000,100000000,100000000]
    plist=[0.00001,0.000001,0.0000001]
    samplelist=[500000000,1000000000,1000000000]


    for i in range(len(plist)):
        p=plist[i]
        sample=samplelist[i]
        circuit=CliffordCircuit(2)
        circuit.set_error_rate(p)
        stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
        stim_circuit=rewrite_stim_code(str(stim_circuit))
        circuit.set_stim_str(stim_circuit)
        circuit.compile_from_stim_circuit_str(stim_circuit)          


        

        new_stim_circuit=circuit.get_stim_circuit()        

        num_shots = sample
        num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)



        error_rate= num_logical_errors / num_shots
        print(f"p: {p}, Sample number: {sample}, Error rate: {error_rate:.9e}")



def repitition_groundTruth2():
    distance = 3

    print("Repetition code ground truth")

    #plist = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    #samplelist = [100000, 500000, 1000000, 100000000, 500000000]

    plist = [0.00001]
    samplelist = [1000000000]
    #plist = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    #samplelist = [100, 50, 10, 100, 500, 100, 100]
    # Open file to store output
    with open("rep_output_log.txt", "w") as logfile:
        
        for i in range(len(plist)):
            p = plist[i]
            sample = samplelist[i]
            print(f"p: {p}, Sample number: {sample}")

            circuit = CliffordCircuit(2)
            circuit.set_error_rate(p)
            stim_circuit = stim.Circuit.generated("repetition_code:memory", rounds=distance*3, distance=distance).flattened()
            stim_circuit = rewrite_stim_code(str(stim_circuit))
            circuit.set_stim_str(stim_circuit)
            circuit.compile_from_stim_circuit_str(stim_circuit)

            new_stim_circuit = circuit.get_stim_circuit()

            num_shots = sample
            num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)

            error_rate = num_logical_errors / num_shots
            output = f"Repetition code: p: {p}, Sample number: {sample}, Error rate: {error_rate:.9e}"
            
            print(output)  # Print to stdout
            logfile.write(output + "\n")  # Write to file


def square_groundTruth2():
    distance = 3

    print("Square code ground truth")

    plist = [0.0001]
    samplelist = [50000000]

    #plist = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    #samplelist = [100, 50, 10, 100, 500, 100, 100]
    # Open file to store output
    with open("square_output_log.txt", "w") as logfile:

        for i in range(len(plist)):
            p = plist[i]
            sample = samplelist[i]

            circuit = CliffordCircuit(2)
            circuit.set_error_rate(p)
            stim_circuit=stim.Circuit(heavy_square_4_circ(p, distance*3, distance)).flattened()
            stim_str=rewrite_stim_code(str(stim_circuit))
            circuit=CliffordCircuit(2)
            circuit.set_error_rate(p)
            circuit.set_stim_str(stim_str)
            circuit.compile_from_stim_circuit_str(stim_str)
            new_stim_circuit=circuit.get_stim_circuit()     

            num_shots = sample
            num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)

            error_rate = num_logical_errors / num_shots
            output = f"Heavy square code: p: {p}, Sample number: {sample}, Error rate: {error_rate:.9e}"
            
            print(output)  # Print to stdout
            logfile.write(output + "\n")  # Write to file



def hexagon_groundTruth2():
    distance = 3

    print("Hexagon code ground truth")

    #plist = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    #samplelist = [100000, 500000, 1000000, 10000000, 10000000]

    plist = [0.00001]
    samplelist = [50000000]

    #plist = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    #samplelist = [100, 50, 10, 100, 500, 100, 100]
    # Open file to store output
    with open("hexagon_output_log.txt", "w") as logfile:

        for i in range(len(plist)):
            p = plist[i]
            sample = samplelist[i]

            circuit = CliffordCircuit(2)
            circuit.set_error_rate(p)
            
            stim_circuit=stim.Circuit(hexagon_3_circ(p, distance*3, distance)).flattened()
            stim_str=rewrite_stim_code(str(stim_circuit))
            circuit=CliffordCircuit(2)
            circuit.set_error_rate(p)
            circuit.set_stim_str(stim_str)
            circuit.compile_from_stim_circuit_str(stim_str)
            new_stim_circuit=circuit.get_stim_circuit()     

            num_shots = sample
            num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)

            error_rate = num_logical_errors / num_shots
            output = f"Hexagon code: p: {p}, Sample number: {sample}, Error rate: {error_rate:.9e}"
            
            print(output)  # Print to stdout
            logfile.write(output + "\n")  # Write to file

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




def Heavy_square():
    print("Heavy square ground truth")
    distance=3
    circuit=CliffordCircuit(2)

    circuit.set_error_rate(0.00005)
    stim_circuit=stim.Circuit(heavy_square_4_circ(0.01, distance*3, distance)).flattened()
    stim_str=rewrite_stim_code(str(stim_circuit))
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.00005)
    circuit.set_stim_str(stim_str)
    circuit.compile_from_stim_circuit_str(stim_str)
    new_stim_circuit=circuit.get_stim_circuit()     


    num_shots = 500
    num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)

    error_rate= num_logical_errors / num_shots
    print(f"Distance: {distance}, Qubit number: {circuit.get_qubit_num()}, Total Measure: {circuit.get_totalMeas()}, Total Noise: {circuit.get_totalnoise()}, Error rate: {error_rate:.9e}")




    distance=5
    stim_circuit=stim.Circuit(heavy_square_4_circ(0.0001, distance*3, distance)).flattened()
    stim_str=rewrite_stim_code(str(stim_circuit))
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.0001)
    circuit.set_stim_str(stim_str)
    circuit.compile_from_stim_circuit_str(stim_str)
    new_stim_circuit=circuit.get_stim_circuit()     


    num_shots = 10000000
    num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)

    error_rate= num_logical_errors / num_shots
    print(f"Distance: {distance}, Qubit number: {circuit.get_qubit_num()}, Total Measure: {circuit.get_totalMeas()}, Total Noise: {circuit.get_totalnoise()}, Error rate: {error_rate:.9e}")


    distance=7
    circuit=CliffordCircuit(2)


    stim_circuit=stim.Circuit(heavy_square_4_circ(0.001, distance*3, distance)).flattened()
    stim_str=rewrite_stim_code(str(stim_circuit))
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.0003)
    circuit.set_stim_str(stim_str)
    circuit.compile_from_stim_circuit_str(stim_str)
    new_stim_circuit=circuit.get_stim_circuit()     


    num_shots =  10000000
    num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)

    error_rate= num_logical_errors / num_shots
    print(f"Distance: {distance}, Qubit number: {circuit.get_qubit_num()}, Total Measure: {circuit.get_totalMeas()}, Total Noise: {circuit.get_totalnoise()}, Error rate: {error_rate:.9e}")




def Heavy_hexagon():
    print("Heavy hexagon ground truth")
    distance=3
    circuit=CliffordCircuit(2)

    circuit.set_error_rate(0.00005)
    stim_circuit=stim.Circuit(hexagon_3_circ(0.00005, distance*3, distance)).flattened()
    stim_str=rewrite_stim_code(str(stim_circuit))
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.00005)
    circuit.set_stim_str(stim_str)
    circuit.compile_from_stim_circuit_str(stim_str)
    new_stim_circuit=circuit.get_stim_circuit()     


    num_shots = 100000000
    num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)

    error_rate= num_logical_errors / num_shots
    print(f"Distance: {distance}, Qubit number: {circuit.get_qubit_num()}, Total Measure: {circuit.get_totalMeas()}, Total Noise: {circuit.get_totalnoise()}, Error rate: {error_rate:.9e}")



    distance=5

    stim_circuit=stim.Circuit(hexagon_3_circ(0.01, distance*3, distance)).flattened()
    stim_str=rewrite_stim_code(str(stim_circuit))
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.0001)
    circuit.set_stim_str(stim_str)
    circuit.compile_from_stim_circuit_str(stim_str)
    new_stim_circuit=circuit.get_stim_circuit()     


    num_shots =10000000
    num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)

    error_rate= num_logical_errors / num_shots
    print(f"Distance: {distance}, Qubit number: {circuit.get_qubit_num()}, Total Measure: {circuit.get_totalMeas()}, Total Noise: {circuit.get_totalnoise()}, Error rate: {error_rate:.9e}")


    distance=7
 
    stim_circuit=stim.Circuit(hexagon_3_circ(0.0003, distance*3, distance)).flattened()
    stim_str=rewrite_stim_code(str(stim_circuit))
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.0003)
    circuit.set_stim_str(stim_str)
    circuit.compile_from_stim_circuit_str(stim_str)
    new_stim_circuit=circuit.get_stim_circuit()     


    num_shots = 10000000
    num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)

    error_rate= num_logical_errors / num_shots
    print(f"Distance: {distance}, Qubit number: {circuit.get_qubit_num()}, Total Measure: {circuit.get_totalMeas()}, Total Noise: {circuit.get_totalnoise()}, Error rate: {error_rate:.9e}")



def find_threshold_binary_search(fitted_func, target_accuracy, low, high, tol=1):
    """
    Uses binary search to find the smallest sample number such that
    fitted_func(sample) < target_accuracy.

    Parameters:
        fitted_func: function that takes sample count and returns accuracy
        target_accuracy: the threshold accuracy percentage (e.g., 5.0)
        low: minimum sample count to check
        high: maximum sample count to check
        tol: tolerance for sample count (defaults to 1)

    Returns:
        threshold sample count
    """
    while low < high:
        print(f"low: {low}, high: {high}")

        mid = (low + high) // 2
        print(fitted_func(mid))
        acc = fitted_func(mid)

        if acc < target_accuracy:
            high = mid
        else:
            low = mid + 1

    return low


from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')  # Disable GUI backend
import matplotlib.pyplot as plt


def stim_samples(distance,p,shots_list,ground_truth,code_type,filename):
    import stim
    circuit = CliffordCircuit(2)
    circuit.set_error_rate(p)


    if code_type == "heavy_square":
        stim_circuit = stim.Circuit(heavy_square_4_circ(p, distance * 3, distance)).flattened()
    elif code_type == "hexagon":
        stim_circuit = stim.Circuit(hexagon_3_circ(p, distance * 3, distance)).flattened()
    else:
        stim_circuit = stim.Circuit.generated(code_type, rounds=distance * 3, distance=distance).flattened()

    stim_circuit = rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit)
    new_stim_circuit = circuit.get_stim_circuit()


    target_accuracy = 5.0  # 5%

    avg_accuracies = []
    std_devs = []

    print("Simulating and calculating accuracies...")
    for shot in shots_list:
        accuracy_list = []
        for _ in range(10):
            num_logical_errors = count_logical_errors(new_stim_circuit, shot)
            error_rate = num_logical_errors / shot
            accuracy = abs((error_rate - ground_truth) / ground_truth) * 100
            accuracy_list.append(accuracy)

        mean_accuracy = np.mean(accuracy_list)
        std_accuracy = np.std(accuracy_list)

        avg_accuracies.append(mean_accuracy)
        std_devs.append(std_accuracy)

        print(f"Shots: {shot}, Mean Error: {mean_accuracy:.4f}%, Std Dev: {std_accuracy:.4f}")

    # Exponential decay fitting: f(x) = A * exp(-alpha * x) + B
    def slow_decay(x, A):
        return A / np.sqrt(x)

    shots_array = np.array(shots_list)
    accuracies_array = np.array(avg_accuracies)
    std_devs_array = np.array(std_devs)

    # Fit with weighting: lower std dev = higher confidence = more weight
    popt, _ = curve_fit(
        slow_decay,
        shots_array,
        accuracies_array,
        p0=(100),
        #sigma=std_devs_array,
        #absolute_sigma=True
    )

    print(f"Fitted parameters: A={popt[0]}")

    fitted_accuracies = slow_decay(shots_array, *popt)

    # Estimate threshold sample number where accuracy < 5%
    threshold_shot = find_threshold_binary_search(
        lambda x: slow_decay(x, *popt),
        target_accuracy,
        low=min(shots_list),
        high=10000 * max(shots_list)  # adjust as needed
    )

    print(f"\nEstimated threshold sample number: {threshold_shot}")

    # Plotting
    fig=plt.figure(figsize=(10, 6))
    plt.errorbar(shots_list, avg_accuracies, yerr=std_devs, fmt='o', label='Mean Error (±1 std)')
    plt.plot(shots_array, fitted_accuracies, linestyle='--', label='Fit the curve by 1/sqrt(x)')
    plt.axhline(y=target_accuracy, color='r', linestyle=':', label='5% MAE Error Threshold')
    if threshold_shot:
        plt.axvline(x=threshold_shot, color='g', linestyle='--', label=f'MAE Threshold Samples ≈ {threshold_shot}')
    plt.xlabel('Number of Shots')
    plt.ylabel('MAE Error (%)')
    plt.title('MAE Error vs. Number of Samples')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename+".png")
    plt.close(fig)





def my_samples(distance,p,shots_list,ground_truth,code_type,filename):
    circuit=CliffordCircuit(3)
    circuit.set_error_rate(p)

    if code_type == "heavy_square":
        stim_circuit = stim.Circuit(heavy_square_4_circ(p, distance * 3, distance)).flattened()
    elif code_type == "hexagon":
        stim_circuit = stim.Circuit(hexagon_3_circ(p, distance * 3, distance)).flattened()
    else:
        stim_circuit = stim.Circuit.generated(code_type, rounds=distance * 3, distance=distance).flattened()

    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit) 


    qubut_num=circuit.get_qubit_num()
    print("Qubit num: ",qubut_num)

    print("Total noise: ",circuit.get_totalnoise())


    sampler=WSampler(circuit)


    sampler.construct_QPEG()

    #lp=sampler.binary_search_zero(1,100,500)
    lp=2
    rp=100

    #rp=sampler.binary_search_half(1,100,600,epsilon=0.03)
    wlist=np.linspace(lp, rp, 12)
    #wlist=list(range(lp, rp+1, 1))

    wlist=[int(x) for x in wlist]
    target_accuracy = 5.0  # 5%

    avg_accuracies = []
    std_devs = []

    for shot in (shots_list):
        accuracy_list = []
        for _ in range(5):

            distribution=sampler.sequential_calc_logical_error_distribution(wlist=wlist,sList=[shot//2]+[max(shot//4,1)]+[max((shot-shot//2-shot//4)//10,1)]*10)
            #distribution=sampler.sequential_calc_logical_error_distribution(wlist=wlist,sList=[shot//4]+[shot//4]+[shot//4]+[(shot//4)//9]*9)
            mu,alpha=sampler.fit_curve(wlist)
            logical_error_rate=sampler.calc_logical_error_rate_by_curve_fitting(lp,mu,alpha)
            accuracy = abs((logical_error_rate - ground_truth) / ground_truth) * 100
            accuracy_list.append(accuracy)

            print(f"Shots: {shot}, Logical Error Rate: {logical_error_rate:.9e}, Error: {accuracy:.4f}%")

        mean_accuracy = np.mean(accuracy_list)
        std_accuracy = np.std(accuracy_list)

        avg_accuracies.append(mean_accuracy)
        std_devs.append(std_accuracy)

        print(f"Shots: {shot}, Mean Error: {mean_accuracy:.4f}%, Std Dev: {std_accuracy:.4f}")


    # Exponential decay fitting: f(x) = A * exp(-alpha * x) + B
    def slow_decay(x, A, alpha):
        return A / (1 + alpha * x)

    shots_array = np.array(shots_list)
    accuracies_array = np.array(avg_accuracies)
    std_devs_array = np.array(std_devs)

    # Fit with weighting: lower std dev = higher confidence = more weight
    popt, _ = curve_fit(
        slow_decay,
        shots_array,
        accuracies_array,
        p0=(100, 0.0005),
        #sigma=std_devs_array,
        #absolute_sigma=True
    )

    print(f"Fitted parameters: A={popt[0]}, alpha={popt[1]}")

    fitted_accuracies = slow_decay(shots_array, *popt)

    # Estimate threshold sample number where accuracy < 5%
    threshold_shot = find_threshold_binary_search(
        lambda x: slow_decay(x, *popt),
        target_accuracy,
        low=min(shots_list),
        high=10000 * max(shots_list)  # adjust as needed
    )

    print(f"\nEstimated threshold sample number: {threshold_shot}")

    # Plotting
    fig=plt.figure(figsize=(10, 6))
    plt.errorbar(shots_list, avg_accuracies, yerr=std_devs, fmt='o', label='Mean Error (±1 std)')
    plt.plot(shots_array, fitted_accuracies, linestyle='--', label='Exponential Fit')
    plt.axhline(y=target_accuracy, color='r', linestyle=':', label='5% Error Threshold')
    if threshold_shot:
        plt.axvline(x=threshold_shot, color='g', linestyle='--', label=f'Threshold Samples≈ {threshold_shot}')
    plt.xlabel('Number of Shots')
    plt.ylabel('Error (%)')
    plt.title('Error vs. Number of Samples')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename+".png")
    plt.close(fig)








if __name__ == "__main__":
    #QEPG_sample()   
    #compare_two_method()
    #stim_sample()
    #print("Surface code ground truth")
    #surface_groundTruth()
    #print("Repitition code ground truth")
    #repitition_groundTruth()
    #print("Color code ground truth")
    
    
    #stim_samples(distance=3,p=0.0001,shots_list=[2500000,2800000,3200000,3600000,4000000,4500000],ground_truth=2.254e-05,code_type="surface_code:rotated_memory_z",filename="surface_code_3_0.0001")


    '''    
    stim_samples(distance=3,p=0.001,shots_list=[10000,20000,50000,80000,100000,200000,500000,1000000,1500000,2000000,2500000,2800000,3200000,3600000,4000000,4500000],ground_truth=6.386e-05,code_type="repetition_code:memory",filename="results/stimRepeatd3.0001")
    '''

    '''
    my_samples(distance=3,p=0.001,shots_list=[100,500,1000,1500,2000,2500,3000,3500,4000,4500,5000],ground_truth=6.386e-05,code_type="repetition_code:memory",filename="results/mymethodRepeatd3.001")
    '''

    '''
    stim_samples(distance=5,p=0.001,shots_list=[10000,20000,50000,80000,100000,200000,500000,1000000,1500000,2000000,2500000,2800000,3200000,3600000,4000000,4500000,5000000,5500000,6000000,7000000,8000000],ground_truth=1.18e-06,code_type="repetition_code:memory",filename="results/stimRepeatd5.0001")
    '''


    '''
    stim_samples(distance=7,p=0.003,shots_list=[10000,20000,50000,80000,100000,200000,500000,1000000,1500000,2000000,2500000,2800000,3200000,3600000,4000000,4500000,5000000,5500000,6000000,7000000,8000000],ground_truth=1.14e-06,code_type="repetition_code:memory",filename="results/stimRepeatd7.003")
    '''

    '''
    my_samples(distance=5,p=0.001,shots_list=[10000,25000,50000,100000,150000,180000,200000],ground_truth=1.18e-06,code_type="repetition_code:memory",filename="results/mymethodRepeatd5.0001")
    '''


    '''
    my_samples(distance=7,p=0.003,shots_list=[50000,12000,15000,20000,25000,30000,35000,40000,45000,50000],ground_truth=1.14e-06,code_type="repetition_code:memory",filename="results/mymethodRepeatd7.003")
    '''


    '''
    stim_samples(distance=3,p=0.0001,shots_list=[10000,20000,50000,80000,100000,200000,500000,1000000,1500000,2000000,2500000,2800000,3200000,3600000,4000000,4500000],ground_truth=2.254e-05,code_type="surface_code:rotated_memory_z",filename="results/surfaced3.0001")
    '''

    '''
    my_samples(distance=3,p=0.0001,shots_list=[1000,1500,2000,2500,3000,3500,4000,4500,5000,6000,8000,10000],ground_truth=2.254e-05,code_type="surface_code:rotated_memory_z",filename="results/mymethodsurfaced3.0001")


    stim_samples(distance=5,p=0.0005,shots_list=[10000,20000,50000,80000,100000,200000,500000,1000000,1500000,2000000,2500000,2800000,3200000,3600000,4000000,4500000,5000000,5500000,6000000,7000000,8000000],ground_truth=6.905e-05,code_type="surface_code:rotated_memory_z",filename="results/surfaced5.0005")
    '''

    '''
    my_samples(distance=5,p=0.0005,shots_list=[2500,5000,10000,15000,25000,30000,35000,40000,50000],ground_truth=6.905e-05,code_type="surface_code:rotated_memory_z",filename="results/mymethodsurfaced5.0005")
    '''


    '''
    stim_samples(distance=7,p=0.0005,shots_list=[10000,20000,50000,80000,100000,200000,500000,1000000,1500000,2000000,2500000,2800000,3200000,3600000,4000000,4500000,5000000,5500000,6000000,7000000,8000000],ground_truth=5.7e-06,code_type="surface_code:rotated_memory_z",filename="results/surfaced7.0005")
    '''

    '''
    my_samples(distance=7,p=0.0005,shots_list=[10000,12000,15000,20000,25000,30000,35000,40000,45000,50000],ground_truth=5.7e-06,code_type="surface_code:rotated_memory_z",filename="results/mymethodsurfaced7.0005")
    '''


    '''
    Heavy_hexagon()
    '''

    '''
    Heavy_square()
    '''

    
    
    #stim_samples(distance=3,p=0.1,shots_list=[10,15,20,25,30,35,50,100,200,400,600,800,1000,1200,1500,2000,2500],ground_truth=5.0027e-01,code_type="surface_code:rotated_memory_z",filename="stimsurface3.1")
    


    #stim_samples(distance=3,p=0.01,shots_list=[10,15,20,25,30,35,50,100,200,400,600,800,1000,1200,1500,2000,2500,3500,4000,5000,5500,6000,6500,7000,7500,8000,8500,9000],ground_truth=1.4942e-01,code_type="surface_code:rotated_memory_z",filename="stimsurface3.01")
    

    #stim_samples(distance=3,p=0.001,shots_list=[500,600,800,1000,1200,1500,2000,2500,3500,4000,5000,5500,6000,6500,7000,7500,8000,8500,9000,13000,15000,20000,25000,30000,35000,40000],ground_truth=2.252e-03,code_type="surface_code:rotated_memory_z",filename="stimsurface3.001")
    #stim_samples(distance=3,p=0.0001,shots_list=[5000,8000,12000,15000,30000,80000,180000,280000,400000,600000,800000,1000000,1500000,2000000,2200000,2600000,3300000],ground_truth=2.29e-05,code_type="surface_code:rotated_memory_z",filename="stimsurface3.0001")


    stim_samples(distance=3,p=0.00001,shots_list=[8000,12000,15000,30000,80000,180000,280000,400000,800000,1000000,1500000,2000000,2600000,3300000,6000000,9000000,12000000,36000000,48000000,64000000,98000000,130000000,260000000],ground_truth=2.32e-07,code_type="surface_code:rotated_memory_z",filename="stimsurface3.00001")
    '''    
    stim_samples(distance=5,p=0.0001,shots_list=[50000,100000,250000,500000,1000000,2200000,4200000,7000000,9000000,10000000],ground_truth=2.2e-06,code_type="hexagon",filename="results/stimhexagon5.0001")
    '''





    '''
    stim_samples(distance=5,p=0.0001,shots_list=[10000,20000,50000,80000,100000,200000,500000,1000000,1500000,2000000,2500000,2800000,3200000,3600000,4000000,4500000],ground_truth=3.4e-06,code_type="heavy_square",filename="results/stimheavysquared5.0001")
    
        
    stim_samples(distance=7,p=0.0003,shots_list=[10000,20000,50000,80000,100000,200000,500000,1000000,1500000,2000000,2500000,2800000,3200000,3600000,4000000,4500000],ground_truth=5.633e-05,code_type="heavy_square",filename="results/stimheavysquared7.0003")
    '''



    
    #my_samples(distance=5,p=0.0001,shots_list=[500,2500,5000,10000,20000,40000],ground_truth=2.2e-06,code_type="hexagon",filename="results/myMethodhexagond5.0001")
    

    #my_samples(distance=3,p=0.00005,shots_list=[500,1500,2500,3500,4500,5000,6000,7000,10000,12000],ground_truth=5.632e-05,code_type="heavy_square",filename="results/myMethodheavysquared3.00005")
    

    
    #surface_groundTruth2()
    



    '''
    stim_samples(distance=3,p=0.00001,shots_list=[350000,520000,1000000,3200000,6200000,9200000,12000000,32000000,62000000],ground_truth=2.32e-07,code_type="surface_code:rotated_memory_z",filename="pdiff/STIMsurfaced30.00001")
    '''


    '''
    my_samples(distance=3,p=0.0001,shots_list=[100,250,500,800,1000,1200,1500,1800,2000],ground_truth=2.29e-05,code_type="surface_code:rotated_memory_z",filename="pdiff/mysurfaced30.0001")
    '''

    '''
    stim_samples(distance=3,p=0.1,shots_list=[10,35,52,100,150,200,250,300,350,500],ground_truth=3.1782e-01,code_type="repetition_code:memory",filename="pdiff/STIMrepetition30.1")
    stim_samples(distance=3,p=0.1,shots_list=[10,35,52,100,150,200,250,300,350,500],ground_truth=5.0042e-01,code_type="heavy_square",filename="pdiff/STIMsquare30.1")
    stim_samples(distance=3,p=0.1,shots_list=[10,35,52,100,150,200,250,300,350,500],ground_truth=4.9515e-01,code_type="hexagon",filename="pdiff/STIMhexagon30.1")


    stim_samples(distance=3,p=0.01,shots_list=[100,350,1000,1500,2000,4000,4500,5200,10000,20000],ground_truth=6.504e-03,code_type="repetition_code:memory",filename="pdiff/STIMrepetition30.01")
    stim_samples(distance=3,p=0.01,shots_list=[10,35,100,150,200,400,450,520,1000,2000],ground_truth=3.53088e-03,code_type="heavy_square",filename="pdiff/STIMsquare30.01")
    stim_samples(distance=3,p=0.01,shots_list=[10,35,100,150,200,400,450,520,1000,2000],ground_truth=7.086e-03,code_type="hexagon",filename="pdiff/STIMhexagon30.01")


    stim_samples(distance=3,p=0.001,shots_list=[3500,5200,10000,32000,62000,92000,120000,320000,620000,920000,1000000],ground_truth=6.7e-05,code_type="repetition_code:memory",filename="pdiff/STIMrepetition30.001")
    stim_samples(distance=3,p=0.001,shots_list=[50,100,250,500,1000,3500,5200,10000,32000,62000,92000],ground_truth=9.235e-03,code_type="heavy_square",filename="pdiff/STIMsquare30.001")
    stim_samples(distance=3,p=0.001,shots_list=[3500,5200,10000,32000,62000,92000,120000,320000,620000,920000,1000000],ground_truth=7.52e-05,code_type="hexagon",filename="pdiff/STIMhexagon30.001")



    stim_samples(distance=3,p=0.0001,shots_list=[3500,5200,10000,32000,62000,92000,120000,320000,620000,920000,1000000],ground_truth=1.525e-04,code_type="heavy_square",filename="pdiff/STIMhexagon30.0001")
    stim_samples(distance=3,p=0.0001,shots_list=[3500,5200,10000,32000,62000,92000,120000,320000,620000,920000,1000000],ground_truth=7.52e-05,code_type="hexagon",filename="pdiff/STIMhexagon30.0001")
    '''

    #hexagon_groundTruth2()
    #square_groundTruth2()
    #repitition_groundTruth2()


   
    
    #stim_samples(distance=3,p=0.0001,shots_list=[350000,520000,1000000,3200000,6200000,9200000,12000000,32000000,62000000],ground_truth=2.32e-07,code_type="repetition_code:memory",filename="pdiff/STIMrepetition30.0001")

    #stim_samples(distance=3,p=0.01,shots_list=[10,35,100,150,200,400,450,520,1000,2000,5000],ground_truth=3.2235e-01,code_type="hexagon",filename="pdiff/STIMhexagon30.01")
    #stim_samples(distance=3,p=0.001,shots_list=[250,500,1000,2000,3500,4000,5200,8000,10000,12000,15000,18000,23000,27000,30000,32000,35000,38000],ground_truth=7.086e-03,code_type="hexagon",filename="pdiff/STIMhexagon30.001")
    #stim_samples(distance=3,p=0.01,shots_list=[10,35,100,150,200,400,450,520,1000,2000],ground_truth=3.53088e-01,code_type="heavy_square",filename="pdiff/STIMsquare30.01")


    #stim_samples(distance=3,p=0.00001,shots_list=[5200,10000,32000,62000,92000,100000,120000,150000,180000,220000,250000,300000,350000,400000,450000,600000,800000,1000000,1200000,1500000],ground_truth=7.58e-06,code_type="heavy_square",filename="pdiff/STIMsquare30.00001")

    #stim_samples(distance=3,p=0.00001,shots_list=[50000,200000,500000,1500000,3000000,4500000,6000000,8000000,10000000],ground_truth=7.58e-06,code_type="heavy_square",filename="pdiff/STIMsquare30.00001")

    #stim_samples(distance=3,p=0.00001,shots_list=[20000,500000,1000000,2500000,5000000,7500000,9000000,12000000,18000000,25000000,36000000,48000000,64000000,96000000],ground_truth=8.8e-07,code_type="hexagon",filename="pdiff/STIMhexagon30.00001")


    #stim_samples(distance=3,p=0.0001,shots_list=[20000,500000,1000000,2500000,5000000,7500000,9000000,12000000,18000000,25000000,36000000,48000000,64000000,96000000],ground_truth=6.9e-07,code_type="repetition_code:memory",filename="pdiff/STIMrepetition30.0001")


    #stim_samples(distance=3,p=0.00001,shots_list=[25000000,20000,500000,1000000,2500000,5000000,7500000,9000000,12000000,18000000,25000000,36000000,48000000,80000000,120000000,240000000,480000000,640000000,960000000],ground_truth=8e-09,code_type="repetition_code:memory",filename="pdiff/STIMrepetition30.00001")


    #stim_samples(distance=3,p=0.00001,shots_list=[25000000,20000,500000,1000000,2500000,5000000,7500000,9000000,12000000,18000000,25000000,36000000,48000000,80000000,120000000,240000000,480000000,640000000,960000000],ground_truth=8e-09,code_type="repetition_code:memory",filename="pdiff/STIMrepetition30.00001")

    #my_samples(distance=3,p=0.0001,shots_list=[1000,1500,2000,2500,3000,3500,4000,4500,5000],ground_truth=1.525e-04,code_type="heavy_square",filename="pdiff/mymethodsquare3.0001")



    #my_samples(distance=3,p=0.01,shots_list=[1000,1500,2000,2500,3000,3500,4000,4500,5000],ground_truth=3.2235e-01,code_type="hexagon",filename="pdiff/mymethodhexagon3.01")
    #square_groundTruth2()