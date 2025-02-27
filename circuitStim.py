import stim
import pymatching
import numpy as np


class repetitionCode():


    def __init__(self,code_distance, error_rate):
        pass






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



def rewrite_stim_code(code: str) -> str:
    """
    Rewrites a Stim program so that each line contains at most one gate or measurement.
    Lines starting with TICK, R, DETECTOR(, and OBSERVABLE_INCLUDE( are kept as-is.
    Multi-target lines for CX, M, and MR are split up.
    """
    lines = code.splitlines()
    output_lines = []

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            # Skip empty lines (optional: you could also preserve them)
            continue

        # Keep lines that we do NOT want to split
        if (stripped_line.startswith("TICK") or
            stripped_line.startswith("DETECTOR(") or
            stripped_line.startswith("QUBIT_COORDS(") or     
            stripped_line.startswith("OBSERVABLE_INCLUDE(")):
            output_lines.append(stripped_line)
            continue

        tokens = stripped_line.split()
        gate = tokens[0]

        # Handle 2-qubit gate lines like "CX 0 1 2 3 4 5 ..."
        if gate == "CX":
            qubits = tokens[1:]
            # Pair up the qubits [q0, q1, q2, q3, ...] => (q0,q1), (q2,q3), ...
            for i in range(0, len(qubits), 2):
                q1, q2 = qubits[i], qubits[i + 1]
                output_lines.append(f"CX {q1} {q2}")

        # Handle multi-qubit measurements "M 1 3 5 ..." => each on its own line
        elif gate == "M":
            qubits = tokens[1:]
            for q in qubits:
                output_lines.append(f"M {q}")


        elif gate == "MX":
            qubits = tokens[1:]
            for q in qubits:
                output_lines.append(f"H {q}")
                output_lines.append(f"M {q}")

        elif gate == "MY":
            qubits = tokens[1:]
            for q in qubits:
                output_lines.append(f"S {q}")
                output_lines.append(f"S {q}")
                output_lines.append(f"S {q}")
                output_lines.append(f"H {q}")                
                output_lines.append(f"M {q}")



        elif gate == "H":
            qubits = tokens[1:]
            for q in qubits:
                output_lines.append(f"H {q}")

        elif gate == "S":
            qubits = tokens[1:]
            for q in qubits:
                output_lines.append(f"S {q}")            

        # Handle multi-qubit measure+reset "MR 1 3 5 ..." => each on its own line
        elif gate == "MR":
            qubits = tokens[1:]
            for q in qubits:
                output_lines.append(f"M {q}")
                output_lines.append(f"R {q}")

        elif gate == "R":
            qubits = tokens[1:]
            for q in qubits:
                output_lines.append(f"R {q}")
        
        elif gate == "RX":
            qubits = tokens[1:]
            for q in qubits:
                output_lines.append(f"R {q}")
                output_lines.append(f"H {q}")                


        else:
            # If there's some other gate we don't specifically handle,
            # keep it as is, or add more logic if needed.
            output_lines.append(stripped_line)

    return "\n".join(output_lines)



def insert_noise_for_h_cx(code: str, p: float) -> str:
    """
    Inserts X_ERROR(p) and Z_ERROR(p) lines immediately before each H or CX line.

    For example:
      H 0
    becomes:
      X_ERROR(p) 0
      Z_ERROR(p) 0
      H 0

    and:
      CX 0 1
    becomes:
      X_ERROR(p) 0
      Z_ERROR(p) 0
      X_ERROR(p) 1
      Z_ERROR(p) 1
      CX 0 1

    Args:
        code: A string containing Stim code, where gates have already been split 
              into single-target (H) or single-pair (CX) lines.
        p: The probability for X_ERROR and Z_ERROR insertions.

    Returns:
        A modified Stim program (string) with inserted noise lines.
    """
    lines = code.splitlines()
    new_lines = []

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            # Skip or preserve empty lines as desired
            new_lines.append(line)
            continue

        tokens = stripped_line.split()
        gate = tokens[0]

        # If it's H gate with one qubit, insert noise lines
        if gate == "H" and len(tokens) == 2:
            qubit = tokens[1]
            new_lines.append(f"X_ERROR({p}) {qubit}")
            new_lines.append(f"Z_ERROR({p}) {qubit}")
            new_lines.append(stripped_line)

        # If it's CX gate with two qubits, insert noise lines
        elif gate == "CX" and len(tokens) == 3:
            cqubit = tokens[1]
            tqubit = tokens[2]
            new_lines.append(f"X_ERROR({p}) {cqubit}")
            new_lines.append(f"Z_ERROR({p}) {cqubit}")
            new_lines.append(f"X_ERROR({p}) {tqubit}")
            new_lines.append(f"Z_ERROR({p}) {tqubit}")
            new_lines.append(stripped_line)

        else:
            # Otherwise, keep the line as is
            new_lines.append(stripped_line)

    return "\n".join(new_lines)



def str_to_circuit(code: str) -> stim.Circuit:
    """
    Converts a string representation of a Stim program into a stim.Circuit object.
    """
    return stim.Circuit(stim_program_text=str)









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

    #print(circuit._circuit)


    sampler = circuit.compile_sampler()
    num_shots = 1

    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)



    # Configure a decoder using the circuit.
    detector_error_model = circuit._circuit.detector_error_model(decompose_errors=True)


    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)


    predictions = matcher.decode_batch(detection_events)



    #print(predictions)


    
    #print(observable_flips)
    # Count the mistakes.

    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    

    print(num_errors)




