from paulitracer import *



'''
Class of quantum error propagation graph
'''
class QEPG:

    def __init__(self,circuit:CliffordCircuit):
        self._circuit = circuit 
        self._tracer=PauliTracer(circuit)

        self._total_meas=self._circuit._totalMeas
        self._total_noise=self._circuit._totalnoise
        self._XerrorMatrix=np.zeros((3*self._total_noise, self._total_meas), dtype=int)
        self._YerrorMatrix=np.zeros((3*self._total_noise, self._total_meas), dtype=int)
        self._ZerrorMatrix=np.zeros((3*self._total_noise, self._total_meas), dtype=int)

    def compute_graph(self):
        for i in range(self._total_noise):
            self._tracer.reset()
            self._tracer.set_noise_type(1)
            self._tracer.prop_all()
            measured_error=self._tracer.get_measuredError()
            for j in range(self._total_meas):
                if measured_error[j]=='X':
                    self.add_x_type_edge(1,i,j)
                elif measured_error[j]=='Y':
                    self.add_y_type_edge(1,i,j)
                elif measured_error[j]=='Z':
                    self.add_z_type_edge(1,i,j)
            self._tracer.reset()    
            self._tracer.set_noise_type(2)
            self._tracer.prop_all()
            measured_error=self._tracer.get_measuredError()
            for j in range(self._total_meas):
                if measured_error[j]=='X':
                    self.add_x_type_edge(2,i,j)
                elif measured_error[j]=='Y':
                    self.add_y_type_edge(2,i,j)
                elif measured_error[j]=='Z':
                    self.add_z_type_edge(2,i,j)
            self._tracer.reset()    
            self._tracer.set_noise_type(3)
            self._tracer.prop_all()
            measured_error=self._tracer.get_measuredError()
            for j in range(self._total_meas):
                if measured_error[j]=='X':
                    self.add_x_type_edge(3,i,j)
                elif measured_error[j]=='Y':
                    self.add_y_type_edge(3,i,j)
                elif measured_error[j]=='Z':
                    self.add_z_type_edge(3,i,j)



    def add_x_type_edge(self, noise_type,a, b):
        if noise_type==1:
            self._XerrorMatrix[a][b]=1
        elif noise_type==2:
            self._XerrorMatrix[a+self._total_noise][b]=1
        elif noise_type==3:
            self._XerrorMatrix[a+2*self._total_noise][b]=1

    def add_y_type_edge(self, noise_type, a, b):
        if noise_type==1:
            self._YerrorMatrix[a][b]=1
        elif noise_type==2:
            self._YerrorMatrix[a+self._total_noise][b]=1
        elif noise_type==3:
            self._YerrorMatrix[a+2*self._total_noise][b]=1


    def add_z_type_edge(self, noise_type,a, b):
        if noise_type==1:
            self._ZerrorMatrix[a][b]=1
        elif noise_type==2:
            self._ZerrorMatrix[a+self._total_noise][b]=1
        elif noise_type==3:
            self._ZerrorMatrix[a+2*self._total_noise][b]=1


    '''
    Sample error and compute the detector value(Parity)
    Return a result of the detected value
    '''
    def sample_error(self, noise_vector:np.array):
        assert len(noise_vector)==3*self._total_noise
        xerror=np.matmul(self._XerrorMatrix, noise_vector)%2
        yerror=np.matmul(self._YerrorMatrix, noise_vector)%2
        zerror=np.matmul(self._ZerrorMatrix, noise_vector)%2
        detectorresult=np.zeros(self._total_meas)
        for i in range(self._total_meas):
            tmpstr=str(xerror[i])+str(yerror[i])+str(zerror[i])
            if tmpstr=='000':
                detectorresult[i]=0
            elif tmpstr=='001':
                detectorresult[i]=0
            elif tmpstr=='010':
                detectorresult[i]=1
            elif tmpstr=='011':
                detectorresult[i]=1
            elif tmpstr=='100':
                detectorresult[i]=1
            elif tmpstr=='101':
                detectorresult[i]=1
            elif tmpstr=='110':
                detectorresult[i]=0
            elif tmpstr=='111':
                detectorresult[i]=0
        return detectorresult



    def matrix(self):
        pass
