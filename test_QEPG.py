from QEPG import *

'''
def test():
    circuit=CliffordCircuit(2)
    circuit.add_cnot(0,1)

    circuit.add_measurement(0)
    circuit.add_measurement(1)
    total_meas =circuit._totalMeas
    parityMatchGroup=[[0,1]]
    circuit.set_parityMatchGroup(parityMatchGroup)

    observable=[0]
    circuit.set_observable(observable)
    circuit.compile_detector_and_observable()


    QEPGraph=QEPG(circuit)


    QEPGraph.compute_graph()
    XerrorMatrix=QEPGraph._XerrorMatrix
    YerrorMatrix=QEPGraph._YerrorMatrix
    detectorMatrix=(XerrorMatrix+YerrorMatrix)%2
    paritymatrix=np.zeros((len(parityMatchGroup)+1,total_meas), dtype='uint8')
    for i in range(len(parityMatchGroup)):
        for j in parityMatchGroup[i]:
            paritymatrix[i][j]=1
    for i in range(len(observable)):
        paritymatrix[len(parityMatchGroup)][observable[i]]=1
    detectorMatrix=np.matmul(paritymatrix,detectorMatrix)
    print(detectorMatrix%2)
    
    
    QEPGraph.backword_graph_construction()
    detectorMatrix2=QEPGraph._detectorMatrix
    print(detectorMatrix2)
'''



def test():
    circuit=CliffordCircuit(4)

    distance=7
    circuit.set_error_rate(0.0001)  
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    print(stim_circuit)
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit)
    


    QEPGraph=QEPG(circuit)


    QEPGraph.compute_graph()
    XerrorMatrix=QEPGraph._XerrorMatrix
    #print("XerrorMatrix:---------------------------------------")
    #print(XerrorMatrix)
    #print("----------------------------------------------------")
    YerrorMatrix=QEPGraph._YerrorMatrix
    #print("YerrorMatrix:---------------------------------------")
    #print(YerrorMatrix)
    #print("----------------------------------------------------")

    detectorMatrix=(XerrorMatrix+YerrorMatrix)%2
    print("Fresh-------------------------")
    print(detectorMatrix%2)


    QEPGraph.backword_graph_construction()
    print("Backward result-------------------------")
    detectorMatrix2=QEPGraph._detectorMatrix%2
    print(detectorMatrix2)

    assert((detectorMatrix==detectorMatrix2).all())


    '''
    paritymatrix=np.zeros((len(parityMatchGroup)+1,total_meas), dtype='uint8')
    for i in range(len(parityMatchGroup)):
        for j in parityMatchGroup[i]:
            paritymatrix[i][j]=1
    for i in range(len(observable)):
        paritymatrix[len(parityMatchGroup)][observable[i]]=1
    print("Parity matrix:")
    print(paritymatrix)    
    detectorMatrix=np.matmul(paritymatrix,detectorMatrix)
    print("Previous-------------------------")
    print(detectorMatrix%2)
    '''


'''
A
'''
def transpile_stim_with_noise_vector(stimString,noise_vector,totalnoise):
    
    lines = stimString.strip().split('\n')

    MindexToLine={}

    current_line_index=0
    current_M_index=0

    total_detector=0
    for line in lines:
        if line.startswith('M'):
            MindexToLine[current_M_index]=current_line_index
            current_M_index+=1
        if line.startswith('DETECTOR'):     
            total_detector+=1
        current_line_index+=1
 


    s = stim.TableauSimulator(seed=0)
    current_noise_index=0

    detector_result=[]
    observableparity=0
    newstimstr=""
    for line in lines:
        if line.startswith('M'):
            qubit_index=int(line.split(' ')[1])
            s.measure(qubit_index)
            newstimstr+=line+"\n"
        if line.startswith('CX'):
            qubit_index1=int(line.split(' ')[1])
            qubit_index2=int(line.split(' ')[2])
            s.cnot(qubit_index1,qubit_index2)
            newstimstr+=line+"\n"
        if line.startswith('H'):
            qubit_index=int(line.split(' ')[1])
            s.h(qubit_index)
            newstimstr+=line+"\n"
        if line.startswith('S'):
            qubit_index=int(line.split(' ')[1])
            s.s(qubit_index)
            newstimstr+=line+"\n"

        if line.startswith('R'):
            split=line.split(' ')
            for i in range(1,len(split)):
                qubit_index=int(split[i])
                s.reset_z(qubit_index)
            newstimstr+=line+"\n"

        if line.startswith('DETECTOR'):
            split=line.split(' ')        
            parity=0
            for i in range(1,len(split)):
                meas=int(split[i][4:-1])
                tmpmeas=s.current_measurement_record()[meas]
                if tmpmeas:
                    parity+=1
            if parity%2==1:
                detector_result.append(1)
            else:
                detector_result.append(0)

        if line.startswith("OBSERVABLE_INCLUDE(0)"):
            split=line.split(' ')        
            for i in range(1,len(split)):
                meas=int(split[i][4:-1])
                tmpmeas=s.current_measurement_record()[meas]
                if tmpmeas:
                    observableparity+=1
            observableparity=observableparity%2
        if line.startswith('DEPOLARIZE1'):
            split=line.split(' ')
            qubit_index1=int(split[1])
            if noise_vector[current_noise_index]==1:
                s.x(qubit_index1)
                newstimstr+="X error "+str(qubit_index1) +"\n"
            elif noise_vector[current_noise_index+totalnoise]==1:
                s.y(qubit_index1)
                newstimstr+="Y error "+str(qubit_index1) +"\n"
            elif noise_vector[current_noise_index+2*totalnoise]==1:
                s.z(qubit_index1)
                newstimstr+="Z error "+str(qubit_index1) +"\n"
            current_noise_index+=1
            if len(split)==3:
                qubit_index2=int(split[2])
                if noise_vector[current_noise_index]==1:
                    s.x(qubit_index2)
                    newstimstr+="X error "+str(qubit_index2) +"\n"
                elif noise_vector[current_noise_index+totalnoise]==1:
                    s.y(qubit_index2)
                    newstimstr+="Y error "+str(qubit_index2) +"\n"
                elif noise_vector[current_noise_index+2*totalnoise]==1:
                    s.z(qubit_index2)
                    newstimstr+="Z error "+str(qubit_index2) +"\n"
                current_noise_index+=1


    print("-----------------------------New stim circuit:---------------------------------")
    print(newstimstr)

    measurement_result=s.current_measurement_record()


    detector_result.append(observableparity)
    return detector_result




'''
Test the correctness of detector matrix with Stim simulator

Idea: Sample a noise, convert the circuit to circuit only with these noise, 
and then compare the result of detector with Stim simulator.
'''
def test_with_stim_tableau():
    circuit=CliffordCircuit(4)

    distance=3
    circuit.set_error_rate(0.0001)  
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=3*distance,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    print(stim_circuit)
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit)
    stimcircuit=circuit.get_stim_circuit()
    print("----------------- Original circuit-----------------------------")
    print(stimcircuit)



    '''
    First step, sample a noise
    '''
    totalnoise=circuit.get_totalnoise()
    print("Total noise: ", totalnoise)


    for W in range(1,int(totalnoise/2)):
        for i in range(0,40):
            random_index=sample_fixed_one_two_three(totalnoise,W)
            noise_vector=np.array([0]*3*totalnoise)
            for i in range(totalnoise):
                if random_index[i]==1:
                    noise_vector[i]=1
                elif random_index[i]==2:
                    noise_vector[i+totalnoise]=1
                elif random_index[i]==3:
                    noise_vector[i+2*totalnoise]=1    

            print("Noise vector: ", noise_vector)



            detector_result=transpile_stim_with_noise_vector(str(stimcircuit),noise_vector,totalnoise)

            print("-------------Detector result from stim: -------------")
            print(detector_result)


            QEPGraph=QEPG(circuit)
                #self._QPEGraph.compute_graph()
            QEPGraph.backword_graph_construction()    


            detectorMatrix=QEPGraph._detectorMatrix%2   
            total_meas=circuit.get_totalMeas()
            observable=circuit.get_observable()
            parity_group=circuit.get_parityMatchGroup()
            paritymatrix=np.zeros((len(parity_group)+1,total_meas), dtype='uint8')

            #print(len(parity_group))
            for i in range(len(parity_group)):
                for j in parity_group[i]:
                    paritymatrix[i][j]=1
            #print("observable: {}".format(observable))
            for i in range(len(observable)):
                paritymatrix[len(parity_group)][observable[i]]=1
            detectorMatrix=np.matmul(paritymatrix,detectorMatrix)




            #print(dectectorMatrix.shape, noise_vector.shape)
            mydetectorresult=np.matmul(detectorMatrix, noise_vector)%2    



            print("-------------My Detector result: -------------")
            print(list(mydetectorresult))

            assert((detector_result==list(mydetectorresult)))





if __name__ == "__main__":

    #test()
    test_with_stim_tableau()