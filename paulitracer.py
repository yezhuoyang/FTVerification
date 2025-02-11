

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
        self._syndromeErrorTable={}
        #Store the repeat match group
        #For example, if we require M0=M1, M2=M3, then the match group is [[0,1],[2,3]]
        self._parityMatchGroup=[]


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





#Trace the pauli frame according to the circuit
class PauliTracer:
    def __init__(self, circuit:CliffordCircuit):
        self._inducedNoise=["I"]*circuit._qubit_num
        self._measuredError={}
        self._circuit=circuit
        self._dataqubits=[i for i in range(circuit._qubit_num)]
        self._syndromequbits=[]
        self._parityMatchGroup=circuit.get_parityMatchGroup()

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
        self._inducedNoise=["I"]*circuit._qubit_num 
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


rep_decoder={"01111":"IXI","00101":"IIX","01010":"XII"}


rep_paritygroup=[[0],[1,2],[3,4]]


steane_decoder={"000001":"XIIIIII","000010":"IXIIIII","000011":"IIXIIII","000100":"IIIXIII","000101":"IIIIXII","000110":"IIIIIXI","000111":"IIIIIIX",
                "001000":"ZIIIIII","010000":"IZIIIII","011000":"IIZIIII","100000":"IIIZIII","101000":"IIIIZII","110000":"IIIIIZI","111000":"IIIIIIZ"}

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
            self._pauliTracer.evolve_all()
            self._finaltableX[i]=(self._pauliTracer.get_measuredError(), self._pauliTracer.get_inducedNoise())
        for i in range(self._totalnoise):
            self._pauliTracer.reset()
            self._pauliTracer.set_noise_type(i, 3)
            self._pauliTracer.evolve_all()
            self._finaltableZ[i]=(self._pauliTracer.get_measuredError(), self._pauliTracer.get_inducedNoise())


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




#Test
if __name__ == "__main__":

    circuit=CliffordCircuit(2)
    circuit.read_circuit_from_file("code/repetition")
    circuit.setShowNoise(True)
    circuit.set_parityMatchGroup(rep_paritygroup)
    tracer=PauliTracer(circuit) 
    tracer.set_dataqubits([1,2,3])



    ftverifier=OneFaultFTVerifier(tracer)
    ftverifier.generate_table()
    ftverifier.transform_table_unique()
    ftverifier.print_unique_table()

    #ftverifier.filter_table()
    #ftverifier.print_filter_table()


    #print(ftverifier.verify_fault_tolerance())
    #print(circuit.get_yquant_latex())
    #tracer=PauliTracer(circuit)
    #tracer.evolve_all()   
    #tracer.print_measuredError()   
    #print(circuit)