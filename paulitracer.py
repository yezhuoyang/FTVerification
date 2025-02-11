

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
    def __init__(self, qubitindex):
        self._name="M"
        self._qubitindex = qubitindex

    def __str__(self):
        return self._name + "[" + str(self._qubitindex) + "]"



#Class: CliffordCircuit
class CliffordCircuit:


    def __init__(self, qubit_num):
        self._qubit_num = qubit_num
        self._totalnoise=0
        self._totalgates=0
        self._gatelists=[]
        self._index_to_noise={}
        self._shownoise=False


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
        self._gatelists.append(Measurement(qubit))

    
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
        self._stabilizers=["I"]*qubit_num
        self._circuit=circuit


    def set_initial_stabilizers(self, stabilizers):
        self._stabilizers=stabilizers


    def print_stabilizers(self):
        print(self._stabilizers)


    def evolve_CNOT(self, control, target):
        pauliStr=self._stabilizers[control]+self._stabilizers[target]
        if pauliStr=="XI":
            self._stabilizers[control]="X"
            self._stabilizers[target]="X"
        elif pauliStr=="XX":
            self._stabilizers[control]="X"
            self._stabilizers[target]="I"   
        elif pauliStr=="IZ":         
            self._stabilizers[control]="Z"
            self._stabilizers[target]="Z"
        elif pauliStr=="ZZ":         
            self._stabilizers[control]="I"
            self._stabilizers[target]="Z"
        elif pauliStr=="IY":
            self._stabilizers[control]="Z"
            self._stabilizers[target]="Y"
        elif pauliStr=="YI":
            self._stabilizers[control]="Y"
            self._stabilizers[target]="X"
        elif pauliStr=="XY":
            self._stabilizers[control]="Y"
            self._stabilizers[target]="Z"
        elif pauliStr=="YX":
            self._stabilizers[control]="Y"
            self._stabilizers[target]="I"
        elif pauliStr=="XZ":
            self._stabilizers[control]="Y"
            self._stabilizers[target]="Y"
        elif pauliStr=="YZ":
            self._stabilizers[control]="X"
            self._stabilizers[target]="Y"
        elif pauliStr=="ZY":
            self._stabilizers[control]="I"
            self._stabilizers[target]="Y"        
                
    def evolve_CZ(self, control, target):
        pass

    def evolve_H(self, qubit):
        if self._stabilizers[qubit]=="X":
            self._stabilizers[qubit]="Z"
        elif self._stabilizers[qubit]=="Z":
            self._stabilizers[qubit]="X"


    def evolve_P(self, qubit):
        if self._stabilizers[qubit]=="X":
            self._stabilizers[qubit]="Y"
        elif self._stabilizers[qubit]=="Y":
            self._stabilizers[qubit]="X"        


    def evolve_X(self, qubit):
        pass


    def evolve_Y(self,qubit):
        pass


    def evolve_Z(self,qubit):
        pass


    def evolve_all(self):
        for gate in self._circuit._gatelists:
            if isinstance(gate, SingeQGate):
                if gate._name=="H":
                    self.evolve_H(gate._qubitindex)
                elif gate._name=="P":
                    self.evolve_P(gate._qubitindex)
                elif gate._name=="X":
                    self.evolve_X(gate._qubitindex)
                elif gate._name=="Y":
                    self.evolve_Y(gate._qubitindex)
                elif gate._name=="Z":
                    self.evolve_Z(gate._qubitindex)
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
            elif isinstance(gate, Measurement):
                pass


#Test
if __name__ == "__main__":
    circuit=CliffordCircuit(3)
    circuit.add_hadamard(0)
    circuit.add_cnot(0,1)
    #circuit.add_cnot(1,2)
    circuit.add_measurement(2)
    circuit.setShowNoise(True)
    print(circuit)

    circuit.set_noise_type(2, 1)
    circuit.set_noise_type(3, 2)
    circuit.reset_noise_type()
    circuit.show_all_noise()


    #tracer=PauliTracer(3, circuit)
    #tracer.set_initial_stabilizers(["X", "I", "I"])
    #tracer.evolve_all()
