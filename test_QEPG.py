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

    distance=5
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




if __name__ == "__main__":

    test()