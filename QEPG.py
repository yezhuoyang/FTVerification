from paulitracer import *



'''
Class of quantum error propagation graph
'''
class QEPG:

    def __init__(self,circuit:CliffordCircuit):
        self._circuit = circuit 
        self._tracer=PauliTracer(circuit)

    def compute_graph(self):
        total_noise=self._circuit._totalnoise
        total_meas=self._circuit._totalMeas
        for i in range(total_noise):
            self._tracer.reset()
            self._tracer.set_noise_type(1)
            self._tracer.prop_all()
            measured_error=self._tracer.get_measuredError()
            for j in range(total_meas):
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
            for j in range(total_meas):
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
            for j in range(total_meas):
                if measured_error[j]=='X':
                    self.add_x_type_edge(3,i,j)
                elif measured_error[j]=='Y':
                    self.add_y_type_edge(3,i,j)
                elif measured_error[j]=='Z':
                    self.add_z_type_edge(3,i,j)



    def add_x_type_edge(self, noise_type,a, b):
        pass


    def add_y_type_edge(self, noise_type a, b):
        pass


    def add_z_type_edge(self, noise_type,a, b):
        pass

    '''
    Sample error and compute the detector value(Parity)
    '''
    def sample_error(self, noise_vector):
        pass


    def matrix(self):
        pass
