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


from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')  # Disable GUI backend
import matplotlib.pyplot as plt



def stim_samples(distance, p, shots_list, ground_truth, code_type, filename):
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

    avg_MAE = []
    std_devs = []

    print("Simulating and calculating accuracies...")
    for shot in shots_list:
        accuracy_list = []
        for _ in range(20):
            num_logical_errors = count_logical_errors(new_stim_circuit, shot)
            error_rate = num_logical_errors / shot
            accuracy = abs((error_rate - ground_truth) / ground_truth) * 100
            accuracy_list.append(accuracy)

        mean_MAE = np.mean(accuracy_list)
        std_MAE = np.std(accuracy_list)

        avg_MAE.append(mean_MAE)
        std_devs.append(std_MAE)

        print(f"Shots: {shot}, Mean Error: {mean_MAE:.4f}%, Std Dev: {std_MAE:.4f}")

    shots_array = np.array(shots_list)
    accuracies_array = np.array(avg_MAE)
    std_devs_array = np.array(std_devs)

    # === First Plot: MAE vs Number of Samples ===
    def slow_decay(x, A):
        return A / np.sqrt(x)

    popt, _ = curve_fit(slow_decay, shots_array, accuracies_array, p0=(100))
    fitted_accuracies = slow_decay(shots_array, *popt)

    fig = plt.figure(figsize=(10, 6))
    plt.errorbar(shots_list, avg_MAE, yerr=std_devs, fmt='o', label='Mean Error (±1 std)')
    plt.plot(shots_array, fitted_accuracies, linestyle='--', label='Fitted Curve A/sqrt(x)')
    plt.xlabel('Number of Shots')
    plt.ylabel('Mean Absolute Error (%)')
    plt.title('MAE vs. Number of Samples for Surface Code(d=3)')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename + ".png")
    plt.close(fig)

    # === Second Plot: MAE vs log(1/m) ===
    # === New Plot: log10(MAE) vs log10(m) ===
    log_shots = np.log10(shots_array)           # log10(m)
    log_mae = np.log10(accuracies_array)        # log10(MAE)

    def linear(x, a, b):
        return a * x + b

    popt_linear, _ = curve_fit(linear, log_shots, log_mae)
    slope, intercept = popt_linear
    print(f"Linear fit: log(MAE) = {slope:.6f} * log(m) + {intercept:.6f}")

    fitted_log_mae = linear(log_shots, *popt_linear)

    fig_loglog = plt.figure(figsize=(10, 6))
    plt.plot(log_shots, log_mae, 'o', label='log(MAE) data')
    plt.plot(log_shots, fitted_log_mae, '--', label=f'Linear Fit: y = {slope:.2f}x + {intercept:.2f}')
    plt.xlabel(r'$\log_{10}(\mathrm{shots})$')
    plt.ylabel(r'$\log_{10}(\mathrm{MAE})$')
    plt.title('log(MAE) vs. log(Number of Samples) for Surface Code(d=3)')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename + "_loglog_plot.png")
    plt.close(fig_loglog)








if __name__ == "__main__":
       stim_samples(distance=3,p=0.001,shots_list=[100,500,1000,2000,5000,10000,15000,20000,40000,50000,70000,80000,100000,120000,150000,180000,200000,250000,300000],ground_truth=2.252e-03,code_type="surface_code:rotated_memory_z",filename="STIMsurfaced3MAE0.001")