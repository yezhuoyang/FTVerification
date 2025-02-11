

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
        self._index_to_noise={}
        self._shownoise=False

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



    def set_noise_type(self, noiseindex, noisetype):
        self._index_to_noise[noiseindex].set_noisetype(noisetype)


    def reset_noise_type(self):
        for i in range(self._totalnoise):
            self._index_to_noise[i].set_noisetype(0)

    def show_all_noise(self):
        for i in range(self._totalnoise):
            print(self._index_to_noise[i])


    def add_cnot(self, control, target):
        self._gatelists.append(pauliNoise(self._totalnoise, control))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1
        self._gatelists.append(pauliNoise(self._totalnoise, target))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1
        self._gatelists.append(TwoQGate(twoQGateindices["CNOT"], control, target))


    def add_hadamard(self, qubit):
        self._gatelists.append(pauliNoise(self._totalnoise, qubit))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1        
        self._gatelists.append(SingeQGate(oneQGateindices["H"], qubit))


    def add_phase(self, qubit):
        self._gatelists.append(pauliNoise(self._totalnoise, qubit))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1      
        self._gatelists.append(SingeQGate(oneQGateindices["P"], qubit))

    def add_cz(self, qubit1, qubit2):
        self._gatelists.append(pauliNoise(self._totalnoise, qubit1))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1
        self._gatelists.append(pauliNoise(self._totalnoise, qubit1))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1
        self._gatelists.append(TwoQGate(twoQGateindices["CZ"], qubit1, qubit2))     


    def add_paulix(self, qubit):
        self._gatelists.append(pauliNoise(self._totalnoise, qubit))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1     
        self._gatelists.append(SingeQGate(oneQGateindices["X"], qubit))


    def add_pauliy(self, qubit):
        self._gatelists.append(pauliNoise(self._totalnoise, qubit))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1    
        self._gatelists.append(SingeQGate(oneQGateindices["Y"], qubit))


    def add_pauliz(self, qubit):
        self._gatelists.append(pauliNoise(self._totalnoise, qubit))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1    
        self._gatelists.append(SingeQGate(oneQGateindices["Z"], qubit))


    def add_measurement(self, qubit):
        self._gatelists.append(pauliNoise(self._totalnoise, qubit))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1   
        self._gatelists.append(Measurement(self._totalMeas,qubit))
        self._totalMeas+=1

    
    def add_reset(self, qubit):
        self._gatelists.append(Reset(qubit))


    def setShowNoise(self, show):
        self._shownoise=show

    def __str__(self):
        str=""
        for gate in self._gatelists:
            if isinstance(gate, pauliNoise) and not self._shownoise:
                continue
            str+=gate.__str__()+"\n"
        return str






#Trace the pauli frame according to the circuit
class PauliTracer:
    def __init__(self, qubit_num, circuit):
        self._inducedNoise=["I"]*qubit_num
        self._measuredError={}
        self._circuit=circuit


    def set_initial_inducedNoise(self, inducedNoise):
        self._inducedNoise=inducedNoise


    def print_inducedNoise(self):
        print(self._inducedNoise)

    def print_measuredError(self):
        for key in self._measuredError:
            print(key, self._measuredError[key],sep=", ")
        print("\n")


    def evolve_CNOT(self, control, target):
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
                
    def evolve_CZ(self, control, target):
        pass

    def evolve_H(self, qubit):
        if self._inducedNoise[qubit]=="X":
            self._inducedNoise[qubit]="Z"
        elif self._inducedNoise[qubit]=="Z":
            self._inducedNoise[qubit]="X"


    def evolve_P(self, qubit):
        if self._inducedNoise[qubit]=="X":
            self._inducedNoise[qubit]="Y"
        elif self._inducedNoise[qubit]=="Y":
            self._inducedNoise[qubit]="X"        


    def append_X(self, qubit):
        if self._inducedNoise[qubit]=="I":
            self._inducedNoise[qubit]="X"
        elif self._inducedNoise[qubit]=="X":
            self._inducedNoise[qubit]="I"
        elif self._inducedNoise[qubit]=="Y":
            self._inducedNoise[qubit]="Z"
        elif self._inducedNoise[qubit]=="Z":
            self._inducedNoise[qubit]="Y"


    def append_Y(self,qubit):
        if self._inducedNoise[qubit]=="I":
            self._inducedNoise[qubit]="Y"
        elif self._inducedNoise[qubit]=="X":
            self._inducedNoise[qubit]="Z"
        elif self._inducedNoise[qubit]=="Y":
            self._inducedNoise[qubit]="I"
        elif self._inducedNoise[qubit]=="Z":
            self._inducedNoise[qubit]="X"
        


    def append_Z(self,qubit):
        if self._inducedNoise[qubit]=="I":
            self._inducedNoise[qubit]="Z"
        elif self._inducedNoise[qubit]=="X":
            self._inducedNoise[qubit]="Y"
        elif self._inducedNoise[qubit]=="Y":
            self._inducedNoise[qubit]="X"
        elif self._inducedNoise[qubit]=="Z":
            self._inducedNoise[qubit]="I"


    def evolve_all(self):
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
                    self.append_X(gate._qubitindex)
                elif gate._noisetype==2:
                    self.append_Y(gate._qubitindex)
                elif gate._noisetype==3:
                    self.append_Z(gate._qubitindex)
            elif isinstance(gate,Reset):
                self._inducedNoise[gate._qubitindex]="I"
            elif isinstance(gate, Measurement):
                self._measuredError[gate._measureindex]=self._inducedNoise[gate._qubitindex]



class OneFaultFTVerifier:

    def __init__(self):
        pass





#Test
if __name__ == "__main__":
    #circuit=CliffordCircuit(2)
    #circuit.add_cnot(0,1)

    #circuit.add_measurement(0)   
    #circuit.add_measurement(1)

    #circuit.setShowNoise(True)
    #print(circuit)
    #circuit.set_noise_type(0, 1)

    #tracer=PauliTracer(2, circuit)
    #tracer.evolve_all()    
    #tracer.print_measuredError()   


    #tracer=PauliTracer(3, circuit)
    #tracer.set_initial_stabilizers(["X", "I", "I"])
    #tracer.evolve_all()
    circuit=CliffordCircuit(2)
    circuit.read_circuit_from_file("code/repetition")
    print(circuit)