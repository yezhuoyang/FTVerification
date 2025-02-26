import stim
import pymatching
import numpy as np



class circuitStim():

    
    def __init__(self, qubit_num):
        self._qubit_num = qubit_num
        self._circuit = stim.Circuit()
        self._sampler=None

    def add_XError(self, qubit, error_rate):
        self._circuit.append("X_ERROR", [qubit], error_rate)


    def add_ZError(self, qubit, error_rate):
        self._circuit.append("Z_ERROR", [qubit], error_rate)


    def add_cnot(self, control, target):
        self._circuit.append("CNOT", [control, target])


    def add_hadamard(self, qubit):
        self._circuit.append("H", [qubit])

    def add_phase(self, qubit):
        self._circuit.append("S", [qubit])


    def add_cz(self, qubit1, qubit2):
        self._circuit.append("CZ", [qubit1, qubit2])

    def add_paulix(self, qubit):
        self._circuit.append("X", [qubit])


    def add_pauliy(self, qubit):
        self._circuit.append("Y", [qubit])


    def add_pauliz(self, qubit):
        self._circuit.append("Z", [qubit])


    def add_measurement(self, qubit):
        self._circuit.append("M", [qubit])


    def add_detector(self):
        self._circuit.append("DETECTOR", [stim.target_rec(-1)])


    def add_reset(self, qubit):
        self._circuit.append("R", [qubit])

    
    def compile_sampler(self):
        self._sampler = self._circuit.compile_detector_sampler()
        return self._sampler


    def add_observable(self):
        self._circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)





if __name__ == "__main__":

    # Create a circuit with 5 qubits
    circuit =circuitStim(5)

    # Add some gates
    circuit.add_cnot(0, 3)
    circuit.add_XError(0, 0.1)
    circuit.add_XError(3, 0.1)
    circuit.add_cnot(1, 3)
    circuit.add_XError(1, 0.1)
    circuit.add_XError(3, 0.1)


    circuit.add_cnot(1, 4)
    circuit.add_XError(1, 0.1)
    circuit.add_XError(4, 0.1)
    circuit.add_cnot(2, 4)
    circuit.add_XError(2, 0.1)
    circuit.add_XError(4, 0.1)

    circuit.add_measurement(3)
    circuit.add_detector()
    circuit.add_measurement(4)
    circuit.add_detector()
    circuit.add_measurement(0)
    circuit.add_observable()

    print(circuit._circuit)


    sampler = circuit.compile_sampler()
    num_shots = 100

    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)


    print(observable_flips)



    # Configure a decoder using the circuit.
    detector_error_model = circuit._circuit.detector_error_model(decompose_errors=True)


    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)


    predictions = matcher.decode_batch(detection_events)


    #print(predictions)


    
    print(observable_flips)
    # Count the mistakes.
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    

    print(num_errors)




