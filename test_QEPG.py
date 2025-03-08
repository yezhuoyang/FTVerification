from QEPG import *


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


if __name__ == "__main__":

    test()