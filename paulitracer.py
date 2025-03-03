import stim
import numpy as np
import pymatching
import math


oneQGate_ = ["H", "P", "X", "Y", "Z"]
oneQGateindices={"H":0, "P":1, "X":2, "Y":3, "Z":4}


twoQGate_ = ["CNOT", "CZ"]
twoQGateindices={"CNOT":0, "CZ":1}

pauliNoise_ = ["I","X", "Y", "Z"]
pauliNoiseindices={"I":0,"X":1, "Y":2, "Z":3}


class SingeQGate:
    def __init__(self, gateindex, qubitindex):
        self._name = oneQGate_[gateindex]
        self._qubitindex = qubitindex

    def __str__(self):
        return self._name + "[" + str(self._qubitindex) + "]"


class TwoQGate:
    def __init__(self, gateindex, control, target):
        self._name = twoQGate_[gateindex]
        self._control = control
        self._target = target

    def __str__(self):
        return self._name + "[" + str(self._control) + "," + str(self._target)+ "]"


class pauliNoise:
    def __init__(self, noiseindex, qubitindex):
        self._name="n"+str(noiseindex)
        self._noiseindex= noiseindex
        self._qubitindex = qubitindex
        self._noisetype=0


    def set_noisetype(self, noisetype):
        self._noisetype=noisetype


    def __str__(self):
        return self._name +"("+pauliNoise_[self._noisetype] +")" +"[" + str(self._qubitindex) + "]"


class Measurement:
    def __init__(self,measureindex ,qubitindex):
        self._name="M"+str(measureindex)
        self._qubitindex = qubitindex
        self._measureindex=measureindex

    def __str__(self):
        return self._name + "[" + str(self._qubitindex) + "]"


class Reset:
    def __init__(self, qubitindex):
        self._name="R"
        self._qubitindex = qubitindex

    def __str__(self):
        return self._name + "[" + str(self._qubitindex) + "]"



#Class: CliffordCircuit
class CliffordCircuit:


    def __init__(self, qubit_num):
        self._qubit_num = qubit_num
        self._totalnoise=0
        self._totalMeas=0
        self._totalgates=0
        self._gatelists=[]
        self._error_rate=0
        self._index_to_noise={}
        self._index_to_measurement={}

        #self._index_to_measurement={}

        self._shownoise=False
        self._syndromeErrorTable={}
        #Store the repeat match group
        #For example, if we require M0=M1, M2=M3, then the match group is [[0,1],[2,3]]
        self._parityMatchGroup=[]
        self._observable=[]
        self._stimcircuit=stim.Circuit()


        #self._error_channel


    def set_error_rate(self, error_rate):
        self._error_rate=error_rate

    def get_stim_circuit(self):
        return self._stimcircuit


    def set_observable(self, observablemeasurements):
        self._observable=observablemeasurements


    def get_observable(self):
        return self._observable


    def set_parityMatchGroup(self, parityMatchGroup):
        self._parityMatchGroup=parityMatchGroup

    def get_parityMatchGroup(self):
        return self._parityMatchGroup

    def get_qubit_num(self):
        return self._qubit_num
    
    def get_totalnoise(self):
        return self._totalnoise

    def get_totalMeas(self):
        return self._totalMeas

    '''
    Read the circuit from a file
    Example of the file:

    NumberOfQubit 6
    cnot 1 2
    cnot 1 3
    cnot 1 0
    M 0
    cnot 1 4
    cnot 2 4
    M 4
    cnot 2 5
    cnot 3 5
    M 5
    R 4
    R 5
    cnot 1 4
    cnot 2 4
    M 4
    cnot 2 5
    cnot 3 5
    M 5

    '''
    def read_circuit_from_file(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                
                if line.startswith("NumberOfQubit"):
                    # Extract the number of qubits
                    self._qubit_num = int(line.split()[1])
                else:
                    # Parse the gate operation
                    parts = line.split()
                    gate_type = parts[0]
                    qubits = list(map(int, parts[1:]))
                    
                    if gate_type == "cnot":
                        self.add_cnot(qubits[0], qubits[1])
                    elif gate_type == "M":
                        self.add_measurement(qubits[0])
                    elif gate_type == "R":
                        self.add_reset(qubits[0])
                    elif gate_type == "H":
                        self.add_hadamard(qubits[0])
                    elif gate_type == "P":
                        self.add_phase(qubits[0])
                    elif gate_type == "CZ":
                        self.add_cz(qubits[0], qubits[1])
                    elif gate_type == "X":
                        self.add_paulix(qubits[0])
                    elif gate_type == "Y":
                        self.add_pauliy(qubits[0])
                    elif gate_type == "Z":
                        self.add_pauliz(qubits[0])
                    else:
                        raise ValueError(f"Unknown gate type: {gate_type}")

    
    '''
    Compile from a stim circuit string.
    '''
    def compile_from_stim_circuit_str(self, stim_str):
        lines = stim_str.splitlines()
        output_lines = []
        maxum_q_index=0
        '''
        First, read and compute the parity match group and the observable
        '''
        parityMatchGroup=[]
        observable=[]

        
        measure_index_to_line={}
        measure_line_to_measure_index={}             
        current_line_index=0
        current_measure_index=0
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                # Skip empty lines (optional: you could also preserve them)
                current_line_index+=1
                continue
            
            # Keep lines that we do NOT want to split
            if (stripped_line.startswith("TICK") or
                stripped_line.startswith("DETECTOR(") or
                stripped_line.startswith("QUBIT_COORDS(") or                
                stripped_line.startswith("OBSERVABLE_INCLUDE(")):
                current_line_index+=1
                continue

            tokens = stripped_line.split()
            gate = tokens[0]

            if gate == "M":
                measure_index_to_line[current_measure_index]=current_line_index
                measure_line_to_measure_index[current_line_index]=current_measure_index
                current_measure_index+=1

            current_line_index+=1
        

        current_line_index=0
        measure_stack=[]
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("DETECTOR("):
                meas_index = [token.strip() for token in stripped_line.split() if token.strip().startswith("rec")]
                meas_index = [int(x[4:-1]) for x in meas_index]
                parityMatchGroup.append([measure_line_to_measure_index[measure_stack[x]] for x in meas_index])
                current_line_index+=1
                continue
            elif stripped_line.startswith("OBSERVABLE_INCLUDE("):
                meas_index = [token.strip() for token in stripped_line.split() if token.strip().startswith("rec")]
                meas_index = [int(x[4:-1]) for x in meas_index]
                observable=[measure_line_to_measure_index[measure_stack[x]] for x in meas_index]
                current_line_index+=1
                continue


            tokens = stripped_line.split()
            gate = tokens[0]
            if gate == "M":
                measure_stack.append(current_line_index)
            current_line_index+=1

        '''
        Insert gates
        '''
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


            if gate == "CX":
                control = int(tokens[1])
                maxum_q_index=maxum_q_index if maxum_q_index>control else control
                target = int(tokens[2])
                maxum_q_index=maxum_q_index if maxum_q_index>target else target
                self.add_cnot(control, target)


            elif gate == "M":
                qubit = int(tokens[1])
                maxum_q_index=maxum_q_index if maxum_q_index>qubit else qubit
                self.add_measurement(qubit)

            elif gate == "H":
                qubit = int(tokens[1])
                maxum_q_index=maxum_q_index if maxum_q_index>qubit else qubit
                self.add_hadamard(qubit)            

            elif gate == "S":
                qubit = int(tokens[1])
                maxum_q_index=maxum_q_index if maxum_q_index>qubit else qubit
                self.add_phase(qubit)    

            
            elif gate == "R":
                qubits = int(tokens[1])
                maxum_q_index=maxum_q_index if maxum_q_index>qubits else qubits
                self.add_reset(qubits)
            
        '''
        Finally, compiler detector and observable
        '''
        self._parityMatchGroup=parityMatchGroup
        self._observable=observable
        self._qubit_num=maxum_q_index+1
        self.compile_detector_and_observable()    




    def save_circuit_to_file(self, filename):
        pass



    def set_noise_type(self, noiseindex, noisetype):
        self._index_to_noise[noiseindex].set_noisetype(noisetype)


    def reset_noise_type(self):
        for i in range(self._totalnoise):
            self._index_to_noise[i].set_noisetype(0)

    def show_all_noise(self):
        for i in range(self._totalnoise):
            print(self._index_to_noise[i])


    def add_cnot(self, control, target):
        self._stimcircuit.append("DEPOLARIZE1", [control], self._error_rate)
        self._gatelists.append(pauliNoise(self._totalnoise, control))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1
        self._gatelists.append(pauliNoise(self._totalnoise, target))
        self._stimcircuit.append("DEPOLARIZE1", [target], self._error_rate)
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1
        self._gatelists.append(TwoQGate(twoQGateindices["CNOT"], control, target))
        self._stimcircuit.append("CNOT", [control, target])


    def add_hadamard(self, qubit):
        self._stimcircuit.append("DEPOLARIZE1", [qubit], self._error_rate)        
        self._gatelists.append(pauliNoise(self._totalnoise, qubit))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1        
        self._gatelists.append(SingeQGate(oneQGateindices["H"], qubit))
        self._stimcircuit.append("H", [qubit])

    def add_phase(self, qubit):
        self._stimcircuit.append("DEPOLARIZE1", [qubit], self._error_rate)   
        self._gatelists.append(pauliNoise(self._totalnoise, qubit))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1      
        self._gatelists.append(SingeQGate(oneQGateindices["P"], qubit))
        self._stimcircuit.append("S", [qubit])

    def add_cz(self, qubit1, qubit2):
        self._gatelists.append(pauliNoise(self._totalnoise, qubit1))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1
        self._gatelists.append(pauliNoise(self._totalnoise, qubit1))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1
        self._gatelists.append(TwoQGate(twoQGateindices["CZ"], qubit1, qubit2))     


    def add_paulix(self, qubit):
        self._stimcircuit.append("DEPOLARIZE1", [qubit], self._error_rate)   
        self._gatelists.append(pauliNoise(self._totalnoise, qubit))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1     
        self._gatelists.append(SingeQGate(oneQGateindices["X"], qubit))
        self._stimcircuit.append("X", [qubit])

    def add_pauliy(self, qubit):
        self._stimcircuit.append("DEPOLARIZE1", [qubit], self._error_rate)  
        self._gatelists.append(pauliNoise(self._totalnoise, qubit))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1    
        self._gatelists.append(SingeQGate(oneQGateindices["Y"], qubit))
        self._stimcircuit.append("Y", [qubit])

    def add_pauliz(self, qubit):
        self._stimcircuit.append("DEPOLARIZE1", [qubit], self._error_rate)  
        self._gatelists.append(pauliNoise(self._totalnoise, qubit))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1    
        self._gatelists.append(SingeQGate(oneQGateindices["Z"], qubit))
        self._stimcircuit.append("Z", [qubit])

    def add_measurement(self, qubit):
        self._stimcircuit.append("DEPOLARIZE1", [qubit], self._error_rate)  
        self._gatelists.append(pauliNoise(self._totalnoise, qubit))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1   
        self._gatelists.append(Measurement(self._totalMeas,qubit))
        self._stimcircuit.append("M", [qubit])
        #self._stimcircuit.append("DETECTOR", [stim.target_rec(-1)])
        self._index_to_measurement[self._totalMeas]=self._gatelists[-1]
        self._totalMeas+=1


    def compile_detector_and_observable(self):
        totalMeas=self._totalMeas
        #print(totalMeas)
        for paritygroup in self._parityMatchGroup:
            #print(paritygroup)
            #print([k-totalMeas for k in paritygroup])
            self._stimcircuit.append("DETECTOR", [stim.target_rec(k-totalMeas) for k in paritygroup])

        self._stimcircuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(k-totalMeas) for k in self._observable], 0)



    def add_reset(self, qubit):
        self._gatelists.append(Reset(qubit))
        self._stimcircuit.append("R", [qubit])

    def setShowNoise(self, show):
        self._shownoise=show

    def __str__(self):
        str=""
        for gate in self._gatelists:
            if isinstance(gate, pauliNoise) and not self._shownoise:
                continue
            str+=gate.__str__()+"\n"
        return str


    def get_yquant_latex(self):
        """
        Convert the circuit (stored in self._gatelists) into a yquant LaTeX string.
        This version simply prints each gate (or noise box) in the order they appear,
        without grouping or any fancy logic.
        """
        lines = []
        # Begin the yquant environment
        lines.append("\\begin{yquant}")
        lines.append("")
        
        # Declare qubits and classical bits.
        # Note: Literal braces in the LaTeX code are escaped by doubling them.
        lines.append("% -- Qubits and classical bits --")
        lines.append("qubit {{$\\ket{{q_{{\\idx}}}}$}} q[{}];".format(self._qubit_num))
        lines.append("cbit {{$c_{{\\idx}} = 0$}} c[{}];".format(self._totalMeas))
        lines.append("")
        lines.append("% -- Circuit Operations --")
        
        # Process each gate in the order they were added.
        for gate in self._gatelists:
            if isinstance(gate, pauliNoise):
                # Print the noise box only if noise output is enabled.
                if self._shownoise:
                    lines.append("[fill=red!80]")
                    # The following format string produces, e.g.,:
                    # "box {$n_{8}$} q[2];"
                    lines.append("box {{$n_{{{}}}$}} q[{}];".format(gate._noiseindex, gate._qubitindex))
            elif isinstance(gate, TwoQGate):
                # Two-qubit gate (e.g., CNOT or CZ).
                if gate._name == "CNOT":
                    # Note: yquant syntax for a CNOT is: cnot q[target] | q[control];
                    line = "cnot q[{}] | q[{}];".format(gate._target, gate._control)
                elif gate._name == "CZ":
                    line = "cz q[{}] | q[{}];".format(gate._target, gate._control)
                lines.append(line)
            elif isinstance(gate, SingeQGate):
                # Single-qubit gate.
                if gate._name == "H":
                    line = "h q[{}];".format(gate._qubitindex)

                lines.append(line)
            elif isinstance(gate, Measurement):
                # Measurement is output as three separate lines.
                lines.append("measure q[{}];".format(gate._qubitindex))
                lines.append("cnot c[{}] | q[{}];".format(gate._measureindex, gate._qubitindex))
                lines.append("discard q[{}];".format(gate._qubitindex))
            elif isinstance(gate, Reset):
                # Reset is output as an initialization command.
                lines.append("init {{$\\ket0$}} q[{}];".format(gate._qubitindex))
            else:
                continue
        
        lines.append("")
        lines.append("\\end{yquant}")
        
        return "\n".join(lines)





import random


def sample_fixed_one_two_three(N, k):
    """
    Returns a list of length N containing exactly k ones 
    (and N-k zeros), in a random order.
    """
    # Step 1: Create a list of k ones and N-k zeros
    arr = [1]*k + [0]*(N-k)
    
    # Step 2: Create a list of 1 or two
    arrtype=[]
    
    for i in range(N):
        arrtype.append(random.randint(1,3))

    
    # Step 2: Shuffle the list randomly
    random.shuffle(arr)
    random.shuffle(arrtype)
    
    return [a * b for a, b in zip(arr, arrtype)]



    



#Trace the pauli frame according to the circuit
#The pauli tracer can propagate pauli error(To verify fault-tolerance) as well as evolve stabilizer(To verify semantic correctness)
class PauliTracer:
    def __init__(self, circuit:CliffordCircuit):
        self._inducedNoise=["I"]*circuit._qubit_num
        self._measuredError={}
        self._circuit=circuit
        self._dataqubits=[i for i in range(circuit._qubit_num)]
        self._syndromequbits=[]
        self._parityMatchGroup=circuit.get_parityMatchGroup()

        #Store the initial stabilizers
        self._initStabilizers=["Z"]*circuit._qubit_num
        self._phasefactor=1

        self._errorrate=circuit._error_rate
        self._stimcircuit=circuit.get_stim_circuit()


    def set_error_rate(self, errorrate):
        self._error_rate=errorrate


    def set_initStabilizers(self, initStabilizers,phasefactor=1):
        self._initStabilizers=initStabilizers
        self._phasefactor=phasefactor


    def get_parityMatchGroup(self):
        return self._parityMatchGroup


    def set_dataqubits(self, dataqubits):
        self._dataqubits=dataqubits
        for i in range(self._circuit._qubit_num):
            if i not in dataqubits:
                self._syndromequbits.append(i)  


    def get_dataqubits(self):
        return self._dataqubits


    def get_inducedNoise(self):
        return self._inducedNoise


    def get_measuredError(self):
        return self._measuredError


    def get_qubit_num(self):
        return self._circuit.get_qubit_num()
    
    def get_totalnoise(self):
        return self._circuit.get_totalnoise()

    def get_totalMeas(self):
        return self._circuit.get_totalMeas()


    def set_noise_type(self, noiseindex, noisetype):
        self._circuit.set_noise_type(noiseindex, noisetype)


    def reset(self):
        self._inducedNoise=["I"]*self._circuit._qubit_num 
        self._measuredError={}      
        self._circuit.reset_noise_type()


    def set_initial_inducedNoise(self, inducedNoise):
        self._inducedNoise=inducedNoise


    def print_inducedNoise(self):
        print(self._inducedNoise)

    def print_measuredError(self):
        for key in self._measuredError:
            print(key, self._measuredError[key],sep=", ")
        print("\n")


    #Propagate pauli noise by CNOT gate
    def prop_CNOT(self, control, target):
        pauliStr=self._inducedNoise[control]+self._inducedNoise[target]
        if pauliStr=="XI":
            self._inducedNoise[control]="X"
            self._inducedNoise[target]="X"
        elif pauliStr=="XX":
            self._inducedNoise[control]="X"
            self._inducedNoise[target]="I"   
        elif pauliStr=="IZ":         
            self._inducedNoise[control]="Z"
            self._inducedNoise[target]="Z"
        elif pauliStr=="ZZ":         
            self._inducedNoise[control]="I"
            self._inducedNoise[target]="Z"
        elif pauliStr=="IY":
            self._inducedNoise[control]="Z"
            self._inducedNoise[target]="Y"
        elif pauliStr=="YI":
            self._inducedNoise[control]="Y"
            self._inducedNoise[target]="X"
        elif pauliStr=="XY":
            self._inducedNoise[control]="Y"
            self._inducedNoise[target]="Z"
        elif pauliStr=="YX":
            self._inducedNoise[control]="Y"
            self._inducedNoise[target]="I"
        elif pauliStr=="XZ":
            self._inducedNoise[control]="Y"
            self._inducedNoise[target]="Y"
        elif pauliStr=="YZ":
            self._inducedNoise[control]="X"
            self._inducedNoise[target]="Y"
        elif pauliStr=="ZY":
            self._inducedNoise[control]="I"
            self._inducedNoise[target]="Y"        

    def evolve_CNOT(self, control, target):
        pass


    #Propagate pauli noise by CZ gate
    def prop_CZ(self, control, target):
        pass

    def evolve_CZ(self, control, target):
        pass


    #Propagate pauli noise by H gate
    def prop_H(self, qubit):
        if self._inducedNoise[qubit]=="X":
            self._inducedNoise[qubit]="Z"
        elif self._inducedNoise[qubit]=="Z":
            self._inducedNoise[qubit]="X"


    def evolve_H(self, qubit):
        pass


    #Propagate pauli noise by P gate
    def prop_P(self, qubit):
        if self._inducedNoise[qubit]=="X":
            self._inducedNoise[qubit]="Y"
        elif self._inducedNoise[qubit]=="Y":
            self._inducedNoise[qubit]="X"        


    def evolve_P(self, qubit):
        pass


    #Add new pauli X noise to the induced noise
    def append_X(self, qubit):
        if self._inducedNoise[qubit]=="I":
            self._inducedNoise[qubit]="X"
        elif self._inducedNoise[qubit]=="X":
            self._inducedNoise[qubit]="I"
        elif self._inducedNoise[qubit]=="Y":
            self._inducedNoise[qubit]="Z"
        elif self._inducedNoise[qubit]=="Z":
            self._inducedNoise[qubit]="Y"


    def evolve_X(self, qubit):
        pass


    #Add new pauli Y noise to the induced noise
    def append_Y(self,qubit):
        if self._inducedNoise[qubit]=="I":
            self._inducedNoise[qubit]="Y"
        elif self._inducedNoise[qubit]=="X":
            self._inducedNoise[qubit]="Z"
        elif self._inducedNoise[qubit]=="Y":
            self._inducedNoise[qubit]="I"
        elif self._inducedNoise[qubit]=="Z":
            self._inducedNoise[qubit]="X"


    def evolve_Y(self, qubit):
        pass


    #Add new pauli Z noise to the induced noise
    def append_Z(self,qubit):
        if self._inducedNoise[qubit]=="I":
            self._inducedNoise[qubit]="Z"
        elif self._inducedNoise[qubit]=="X":
            self._inducedNoise[qubit]="Y"
        elif self._inducedNoise[qubit]=="Y":
            self._inducedNoise[qubit]="X"
        elif self._inducedNoise[qubit]=="Z":
            self._inducedNoise[qubit]="I"


    def evolve_Z(self, qubit):
        pass

    def prop_all(self):
        for gate in self._circuit._gatelists:
            if isinstance(gate, SingeQGate):
                if gate._name=="H":
                    self.prop_H(gate._qubitindex)
                elif gate._name=="P":
                    self.prop_P(gate._qubitindex)
            elif isinstance(gate, TwoQGate):
                if gate._name=="CNOT":
                    self.prop_CNOT(gate._control, gate._target)
                elif gate._name=="CZ":
                    self.prop_CZ(gate._control, gate._target)
            elif isinstance(gate, pauliNoise):
                if gate._noisetype==1:
                    self.append_X(gate._qubitindex)
                elif gate._noisetype==2:
                    self.append_Y(gate._qubitindex)
                elif gate._noisetype==3:
                    self.append_Z(gate._qubitindex)               
            elif isinstance(gate,Reset):
                self._inducedNoise[gate._qubitindex]="I"
            elif isinstance(gate, Measurement):
                self._measuredError[gate._measureindex]=self._inducedNoise[gate._qubitindex]


    def get_stim_circuit(self):
        return self._stimcircuit


    def evolve_stabilizer(self):
        for gate in self._circuit._gatelists:
            if isinstance(gate, SingeQGate):
                if gate._name=="H":
                    self.evolve_H(gate._qubitindex)
                elif gate._name=="P":
                    self.evolve_P(gate._qubitindex)
            elif isinstance(gate, TwoQGate):
                if gate._name=="CNOT":
                    self.evolve_CNOT(gate._control, gate._target)
                elif gate._name=="CZ":
                    self.evolve_CZ(gate._control, gate._target)
            elif isinstance(gate, pauliNoise):
                if gate._noisetype==1:
                    self.evolve_X(gate._qubitindex)
                elif gate._noisetype==2:
                    self.evolve_Y(gate._qubitindex)
                elif gate._noisetype==3:
                    self.evolve_Z(gate._qubitindex)
            elif isinstance(gate,Reset):
                continue
            elif isinstance(gate, Measurement):
                continue






class NaiveSampler():
    def __init__(self, circuit:CliffordCircuit):
        self._qubit_num=circuit._qubit_num     
        self._circuit=circuit
        self._totalnoise=circuit.get_totalnoise()

        self._logical_error_rate=0

        self._dataqubits=None
        self._syndromequbits=None
        self._stimcircuit=None

        self._shots=10

    def set_shots(self, shots):
        self._shots=shots


    def calc_logical_error_rate(self):
        self._stimcircuit=self._circuit.get_stim_circuit()

        sampler = self._stimcircuit.compile_detector_sampler()
       
        detector_error_model = self._stimcircuit.detector_error_model(decompose_errors=True)

        num_errors = 0

        for i in range(self._shots):
            detection_events, observable_flips = sampler.sample(1, separate_observables=True)
            #detection_events, observable_flips = sampler.sample(self._shots, separate_observables=True)
            # Configure a decoder using the circuit.
            #print(detection_events)
            #print(observable_flips)
            
            
            matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
            predictions = matcher.decode_batch(detection_events)
            #print(predictions)
    
        
            actual_for_shot = observable_flips[0]
            predicted_for_shot = predictions[0]
            if not np.array_equal(actual_for_shot, predicted_for_shot):
                num_errors += 1
        
        self._logical_error_rate=num_errors/self._shots

        return self._logical_error_rate


    





rep_decoder={"01111":"IXI","00101":"IIX","01010":"XII"}


rep_paritygroup=[[0],[1,2],[3,4]]


steane_decoder={"000001":"XIIIIII","000010":"IXIIIII","000011":"IIXIIII","000100":"IIIXIII","000101":"IIIIXII","000110":"IIIIIXI","000111":"IIIIIIX",
                "001000":"ZIIIIII","010000":"IZIIIII","011000":"IIZIIII","100000":"IIIZIII","101000":"IIIIZII","110000":"IIIIIZI","111000":"IIIIIIZ"}


steane_logicalZ="ZZZZZZZ"
steane_stabilizers=["IIIXXXX","IXXIIXX","XIXIXIX","IIIZZZZ","IZZIIZZ","ZIZIZIZ"]

steane_paritygroup=[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]]



def filter_string(s, indices):
    return ''.join(s[i] for i in indices)


class OneFaultFTVerifier:

    def __init__(self,pauliTracer:PauliTracer):
        self._pauliTracer=pauliTracer
        self._totalnoise=pauliTracer.get_totalnoise()
        self._totalMeas=pauliTracer.get_totalMeas()
        self._finaltableX={}
        self._finaltableZ={}       
        self._syndromeErrorTable={}
        self._filterdsyndromeErrorTable={}
        self._dataqubits=pauliTracer.get_dataqubits()
        self._parityMatchGroup=pauliTracer.get_parityMatchGroup()


    def generate_table(self):
        for i in range(self._totalnoise):
            self._pauliTracer.reset()
            self._pauliTracer.set_noise_type(i, 1)
            self._pauliTracer.prop_all()
            self._finaltableX[i]=(self._pauliTracer.get_measuredError(), self._pauliTracer.get_inducedNoise())
        for i in range(self._totalnoise):
            self._pauliTracer.reset()
            self._pauliTracer.set_noise_type(i, 3)
            self._pauliTracer.prop_all()
            self._finaltableZ[i]=(self._pauliTracer.get_measuredError(), self._pauliTracer.get_inducedNoise())

    #Transform the table from a dictionary to with many values to a dictionary with two values, just syndrome and the inducde noise
    #Use a bit string of 0,1 to represent the syndrome and a bit string of X,Y,Z to represent the induced noise 
    def transform_table(self,measuredError,inducedNoise):
        table={}
        for i in range(self._totalMeas):
            if measuredError[i]=="X":
                table[i]=1
            elif measuredError[i]=="Y":
                table[i]=1
            else:
                table[i]=0
        inducedNoiseStr=""
        for i in range(self._pauliTracer.get_qubit_num()):  
            if i in self._dataqubits:
                inducedNoiseStr+=inducedNoise[i]
        table["inducedNoise"]=inducedNoiseStr
        return table

    
    def transform_table_unique(self):
        for i in range(self._totalnoise):
            tmptable=self.transform_table(self._finaltableX[i][0], self._finaltableX[i][1])
            syndromeString=""
            for j in range(self._totalMeas):
                syndromeString+=str(tmptable[j])

            self._syndromeErrorTable["n"+str(i)+":X"]=(syndromeString,tmptable["inducedNoise"])

            tmptable=self.transform_table(self._finaltableZ[i][0], self._finaltableZ[i][1])
            syndromeString=""
            for j in range(self._totalMeas):
                syndromeString+=str(tmptable[j])

            self._syndromeErrorTable["n"+str(i)+":Z"]=(syndromeString,tmptable["inducedNoise"])



    def print_unique_table(self):
        print("Unique table")

        for key in self._syndromeErrorTable:
            print(key+"   ", self._syndromeErrorTable[key][0]+"   ", self._syndromeErrorTable[key][1])

        print("\n") 




    def filter_string_by_parity(self, syndromeString):
        '''
        Filter the syndrome string according to the parity match group
        '''
        bitlist=[int(x) for x in syndromeString]
        for group in self._parityMatchGroup:
            parity = sum(bitlist[i] for i in group) % 2
            if parity == 1:
                return False
        return True


    '''
    Filter the table based on the parity match group
    '''
    def filter_table(self):
        for key in self._syndromeErrorTable:
            if self.filter_string_by_parity(self._syndromeErrorTable[key][0]):
                self._filterdsyndromeErrorTable[key]=self._syndromeErrorTable[key]


    def print_filter_table(self):
        print("Filtered table")

        for key in self._filterdsyndromeErrorTable:
            print(key, self._filterdsyndromeErrorTable[key][0], self._filterdsyndromeErrorTable[key][1])

        print("\n")


    def verify_fault_tolerance(self):
        for key in self._filterdsyndromeErrorTable:
            correction=rep_decoder[self._filterdsyndromeErrorTable[key][0]]
            error=self._filterdsyndromeErrorTable[key][1]
            if error!=correction:
                return False
        return True



    def print_table(self):
        print("X error table")
        for i in range(self._totalnoise):
            print("n"+str(i)+"=X", self.transform_table(self._finaltableX[i][0], self._finaltableX[i][1]))

        print("\nZ error table")
        for i in range(self._totalnoise):
            print("n"+str(i)+"=Z", self.transform_table(self._finaltableZ[i][0], self._finaltableZ[i][1]))

        print("\n")




class repetitionCode():


    def __init__(self, code_distance, error_rate):
        self._circuit=CliffordCircuit()





class StimToCliffordCircuit():


    def __init__(self, stimcircuit:stim.Circuit):
        self._stimcircuit=stimcircuit


    def convert(self):
        pass




from circuitStim import rewrite_stim_code



#Test
if __name__ == "__main__":

    #circuit=CliffordCircuit(2)
    #circuit.read_circuit_from_file("code/repetition")
    #circuit.setShowNoise(True)
    #circuit.set_parityMatchGroup(rep_paritygroup)
    #tracer=PauliTracer(circuit) 
    #tracer.set_dataqubits([1,2,3])



    #ftverifier=OneFaultFTVerifier(tracer)
    #ftverifier.generate_table()
    #ftverifier.transform_table_unique()
    #ftverifier.print_unique_table()

    #ftverifier.filter_table()
    #ftverifier.print_filter_table()


    #print(ftverifier.verify_fault_tolerance())
    #print(circuit.get_yquant_latex())
    #tracer=PauliTracer(circuit)
    #tracer.evolve_all()   
    #tracer.print_measuredError()   
    #print(circuit)

    '''
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.001)
    circuit.read_circuit_from_file("code/repetition")

    parityMatchGroup=[[0,2],[1,3],[4]]
    circuit.set_parityMatchGroup(parityMatchGroup)


    observable=[0]
    circuit.set_observable(observable)
    circuit.compile_detector_and_observable()



    
    tracer=PauliTracer(circuit) 
    tracer.set_dataqubits([1,2,3])
    sampler=WSampler(circuit)
    sampler.set_shots(200)
    sampler.set_dataqubits([1,2,3])
    sampler.construct_detector_model()

    sampler.calc_logical_error_rate()
    print(sampler._logical_error_distribution)

    print(sampler._logical_error_rate)
    


    Nsampler=NaiveSampler(circuit)
    Nsampler.set_shots(100000)
    Nsampler.calc_logical_error_rate()
    print(Nsampler._logical_error_rate)
    '''

    stim_circuit=stim.Circuit.generated("repetition_code:memory",rounds=1,distance=3).flattened()
    stim_str=rewrite_stim_code(str(stim_circuit))

    #print(stim_str)

    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.1)
    circuit.compile_from_stim_circuit_str(stim_str)


    #circuit_stim=circuit._stimcircuit


    
    Nsampler=NaiveSampler(circuit)
    Nsampler.set_shots(10000)
    Nsampler.calc_logical_error_rate()
    print(Nsampler._logical_error_rate)

    print("---------------------------------------------------------------")
    #print(circuit._stimcircuit)


    print(circuit._qubit_num)
    
    tracer=PauliTracer(circuit) 
    sampler=WSampler(circuit)
    sampler.set_shots(5000)
    sampler.construct_detector_model()

    sampler.calc_logical_error_rate()
    print(sampler._logical_error_distribution)
    print(sampler._logical_error_rate)
    
    


    #sampler.sample_Xnoise(1)
    
    #print(sampler.has_logical_error())


    #sampler.calc_error_rate_with_fixed_weight(1,10)
    #error_rate=sampler.calc_error_rate_with_fixed_weight(1, 1000)
    

    #print(sampler._logical_error_distribution)
    #print(error_rate)
    #print(observable_flips)
    # Count the mistakes.

