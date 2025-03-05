from paulitracer import *
from multiprocessing import Process, Pool
import os
import signal
from multiprocessing import shared_memory


'''
Class of quantum error propagation graph
'''
class QEPG:

    def __init__(self,circuit:CliffordCircuit):
        self._circuit = circuit 
        self._tracer=PauliTracer(circuit)

        self._total_meas=self._circuit._totalMeas
        self._total_noise=self._circuit._totalnoise
        self._XerrorMatrix=np.zeros((self._total_meas,3*self._total_noise), dtype=int)
        self._YerrorMatrix=np.zeros((self._total_meas,3*self._total_noise), dtype=int)
        self._ZerrorMatrix=np.zeros((self._total_meas,3*self._total_noise), dtype=int)


    def compute_graph(self):
        for i in range(self._total_noise):
            self._tracer.reset()
            self._tracer.set_noise_type(i,1)
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
            self._tracer.set_noise_type(i,2)
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
            self._tracer.set_noise_type(i,3)
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
            self._XerrorMatrix[b][a]=1
        elif noise_type==2:
            self._XerrorMatrix[b][a+self._total_noise]=1
        elif noise_type==3:
            self._XerrorMatrix[b][a+2*self._total_noise]=1

    def add_y_type_edge(self, noise_type, a, b):
        if noise_type==1:
            self._YerrorMatrix[b][a]=1
        elif noise_type==2:
            self._YerrorMatrix[b][a+self._total_noise]=1
        elif noise_type==3:
            self._YerrorMatrix[b][a+2*self._total_noise]=1


    def add_z_type_edge(self, noise_type,a, b):
        if noise_type==1:
            self._ZerrorMatrix[b][a]=1
        elif noise_type==2:
            self._ZerrorMatrix[b][a+self._total_noise]=1
        elif noise_type==3:
            self._ZerrorMatrix[b][a+2*self._total_noise]=1


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


    

#Sample noise with weight K
def sample_noise_and_calc_result(shots,totalnoise,total_meas,W,dtype,shm_detec_name,parity_group, observable):
    shm_detec = shared_memory.SharedMemory(name=shm_detec_name)
    dectectorMatrix = np.ndarray((total_meas,3*totalnoise), dtype=dtype, buffer=shm_detec.buf)  
    detection_events=[]
    observable_flips=[]
    for i in range(shots):
        random_index=sample_fixed_one_two_three(totalnoise,W)
        noise_vector=np.array([0]*3*totalnoise)
        for i in range(totalnoise):
            if random_index[i]==1:
                noise_vector[i]=1
            elif random_index[i]==2:
                noise_vector[i+totalnoise]=1
            elif random_index[i]==3:
                noise_vector[i+2*totalnoise]=1           

        detectorresult=np.matmul(dectectorMatrix, noise_vector)%2

        tmp_detection_events=[]
        for group in parity_group:
            parity=0
            for i in group:
                if detectorresult[i]==1:
                    parity+=1
            parity=parity%2
            if parity==1:
                tmp_detection_events.append(True)
            else:
                tmp_detection_events.append(False)
                
        tmp_observable_flips=[]
        parity=0
        for index in observable:
            if detectorresult[index]==1:
                parity+=1
        parity=parity%2
        if parity==1:
            tmp_observable_flips.append(1)
        else:
            tmp_observable_flips.append(0)
        detection_events.append(tmp_detection_events)
        observable_flips.append(tmp_observable_flips)
    return detection_events, observable_flips





def init_worker():
    # Ignore SIGINT in worker processes so that only the main process catches it.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class WSampler():
    def __init__(self, circuit:CliffordCircuit,distance=3):
        self._inducedNoise=["I"]*circuit._qubit_num
        self._measuredError={}
        self._qubit_num=circuit._qubit_num     
        self._circuit=circuit
        self._totalnoise=circuit.get_totalnoise()

        self._logical_error_distribution=[0]*self._totalnoise
        self._logical_error_rate=0

        
        self._detection_events=[]
        self._observable_flips=[]
        self._dataqubits=None
        self._syndromequbits=None
        self._stimcircuit=None

        self._detector_error_model=None
        self._shots=10

        self._noise_vector=np.array([0]*3*self._totalnoise)


        self._QPEGraph=None

        self._distance=distance

        self._binomial_weights=[0]*self._totalnoise
        self.calc_binomial_weight()


        self._maxvariance=1e-8



    def construct_QPEG(self):
        self._QPEGraph=QEPG(self._circuit)
        self._QPEGraph.compute_graph()


    def set_shots(self, shots):
        self._shots=shots


    def set_dataqubits(self, dataqubits):
        self._dataqubits=dataqubits

    #Sample noise with weight K
    def sample_noise(self,W):
        random_index=sample_fixed_one_two_three(self._totalnoise,W)

        self._noise_vector=np.array([0]*3*self._totalnoise)

        for i in range(self._totalnoise):
            if random_index[i]==1:
                self._noise_vector[i]=1
            elif random_index[i]==2:
                self._noise_vector[i+self._totalnoise]=1
            elif random_index[i]==3:
                self._noise_vector[i+2*self._totalnoise]=1               


    #Propagate the error, and get sample result
    def calc_sample_result(self):
        detectorresult=self._QPEGraph.sample_error(self._noise_vector) 


        tmp_detection_events=[]
        parity_group=self._circuit.get_parityMatchGroup()
        for group in parity_group:
            parity=0
            for i in group:
                if detectorresult[i]==1:
                    parity+=1
            parity=parity%2
            if parity==1:
                tmp_detection_events.append(True)
            else:
                tmp_detection_events.append(False)
                
        observable=self._circuit.get_observable()
        tmp_observable_flips=[]
        parity=0
        for index in observable:
            if detectorresult[index]==1:
                parity+=1
        parity=parity%2
        if parity==1:
            tmp_observable_flips.append(True)
        else:
            tmp_observable_flips.append(False)

        return [tmp_detection_events], [tmp_observable_flips]
         

    def construct_detector_model(self):
        self._stimcircuit=self._circuit.get_stim_circuit()
        self._detector_error_model= self._stimcircuit.detector_error_model(decompose_errors=True)
       


    #Propagate the error, and return if there is a logical error
    def has_logical_error(self):
        # Configure a decoder using the circuit.
        detection_events, observable_flips=self.calc_sample_result()
        #print(detection_events)
        #print(observable_flips)

        matcher = pymatching.Matching.from_detector_error_model(self._detector_error_model)
        predictions = matcher.decode_batch(detection_events)

        if not np.array_equal(observable_flips[0],predictions[0]):
            return True 
        return False





    def binomial_weight(self, W):
        p=self._circuit._error_rate
        N=self._totalnoise
        if N<200:
            return math.comb(N, W) * (p**W) * ((1 - p)**(N - W))
        else:
            lam = N * p
            # PMF(X=W) = e^-lam * lam^W / W!
            # Evaluate in logs to avoid overflow for large W, then exponentiate
            log_pmf = (-lam) + W*math.log(lam) - math.lgamma(W+1)
            return math.exp(log_pmf)


    def calc_binomial_weight(self):
        for i in range(self._totalnoise):
            self._binomial_weights[i]=self.binomial_weight(i)
        


    def calc_logical_error_rate(self):


        exp_noise=int(self._totalnoise*self._circuit._error_rate)
        min_W=max(0,exp_noise-50)
        max_W=min(self._totalnoise,exp_noise+50)

        '''
        for i in range(self._totalnoise):
            if(self._binomial_weights[i]>1e-12):
                maxW=i
                continue
            else:
                break

        if maxW<=30:
            maxW=30
        '''

        total_noise=self._totalnoise
        parity_group=self._circuit.get_parityMatchGroup()
        observable=self._circuit.get_observable()
        QEPGgraph=self._QPEGraph
        total_meas=self._circuit._totalMeas

        XerrorMatrix=QEPGgraph._XerrorMatrix
        YerrorMatrix=QEPGgraph._YerrorMatrix
        detectorMatrix=(XerrorMatrix+YerrorMatrix)%2


        shm_dec = shared_memory.SharedMemory(create=True, size=detectorMatrix.nbytes)
        # Create a NumPy array backed by the shared memory
        shared_array = np.ndarray(detectorMatrix.shape, dtype=detectorMatrix.dtype, buffer=shm_dec.buf)
        # Copy the data into shared memory
        shared_array[:] = detectorMatrix[:]    



        inputs=[]
        for i in range(max_W-min_W+1):
            inputs=inputs+[(self._shots,total_noise,total_meas,i,XerrorMatrix.dtype,shm_dec.name,parity_group, observable)]
        
        
        pool = Pool(processes=os.cpu_count(), initializer=init_worker)

        try:
            # starmap is a blocking call that collects results from each process.
            results = pool.starmap(sample_noise_and_calc_result, inputs)
        except KeyboardInterrupt:
            # Handle Ctrl-C gracefully.
            print("KeyboardInterrupt received. Terminating pool...")
            pool.terminate()
            pool.join()
            return  # or re-raise if you want to propagate the exception


        # 'results' is a list of lists, e.g., [[1, 10, 100], [2, 20, 200], ...]
        # You can concatenate them using a list comprehension or itertools.chain.
        detections=np.array([item for result in results for item in result[0]])
        #detections = np.array([result[0] for result in results])
        #observables = [result[1] for result in results]
        observables=np.array([item for result in results for item in result[1]])       

        detector_error_model = self._circuit._stimcircuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(detector_error_model)


        predictions = matcher.decode_batch(detections)

        flattened_predictions = [item for sublist in predictions for item in sublist]
        flattened_observables = [item for sublist in observables for item in sublist]   



        for i in range(0,max_W-min_W+1):
            tmp_flattened_predictions= flattened_predictions[i*self._shots:(i+1)*self._shots]
            tmp_flattened_observables= flattened_observables[i*self._shots:(i+1)*self._shots]
            errorshots = sum(1 for a, b in zip(tmp_flattened_predictions, tmp_flattened_observables) if a != b)
            self._logical_error_distribution[min_W+i]=errorshots/self._shots   
            self._logical_error_rate+=self._binomial_weights[min_W+i]*self._logical_error_distribution[min_W+i]

             
        
        #print(f"------------Error distribution--------Total Number error: {self._totalnoise}-------------")
        #print(self._logical_error_distribution)
        #print("------------Binomial weights---------------------")
        #print(self._binomial_weights)
        return self._logical_error_rate




def test_QEPG():
    circuit=CliffordCircuit(2)
    circuit.add_cnot(0,1)

    circuit.add_measurement(0)
    circuit.add_measurement(1)
    
    graph=QEPG(circuit)
    graph.compute_graph()
    
    noise_vector=np.array([0]*12)    
    noise_vector[0]=1

    measureresult=graph.sample_error(noise_vector)

    assert  np.array_equal(measureresult, [1,1])




if __name__ == "__main__":
    '''
    circuit=CliffordCircuit(2)
    circuit.add_cnot(0,1)


    circuit.add_measurement(0)
    circuit.add_measurement(1)
    print(circuit._totalnoise)


    graph=QEPG(circuit)


    graph.compute_graph()
    print(graph._XerrorMatrix)
    print(graph._YerrorMatrix)
    print(graph._ZerrorMatrix)
    '''


    '''
    stim_circuit=stim.Circuit.generated("repetition_code:memory",rounds=1,distance=3).flattened()
    stim_str=rewrite_stim_code(str(stim_circuit))

    #print(stim_str)

    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.005)
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
    sampler.set_shots(2000)
    sampler.construct_detector_model()

    sampler.calc_logical_error_rate()
    print(sampler._logical_error_distribution)
    print(sampler._logical_error_rate)
    '''


    '''
    stim_circuit=stim.Circuit.generated("repetition_code:memory",rounds=15,distance=30).flattened()
    stim_str=rewrite_stim_code(str(stim_circuit))

    #print(stim_str)

    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.2)
    circuit.compile_from_stim_circuit_str(stim_str)



    detector_error_model = circuit._stimcircuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)



    QEPGgraph=QEPG(circuit)
    QEPGgraph.compute_graph()


    total_noise=circuit._totalnoise
    print(total_noise)
    W=5
    parity_group=circuit.get_parityMatchGroup()
    observable=circuit.get_observable()


    inputs = [(total_noise,W,QEPGgraph, parity_group, observable) for _ in range(10)]
    
    # Create a pool of worker processes
    with Pool(processes=40) as pool:
        # pool.map will gather the output from each call to `worker`
        results = pool.starmap(sample_noise_and_calc_result, inputs)

    # 'results' is a list of lists, e.g., [[1, 10, 100], [2, 20, 200], ...]
    # You can concatenate them using a list comprehension or itertools.chain.
    detections = [result[0] for result in results]
    detections=np.array(detections)

    observables = [result[1] for result in results]

    print("-------------------------------------------")
    print(detections)
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print(detections.shape)


    predictions = matcher.decode_batch(detections)


    print(predictions)
    print("-------------------------------------------")   
    print(observables)


    flattened_predictions = [item for sublist in predictions for item in sublist]
    flattened_observables = [item for sublist in observables for item in sublist]   

    not_matching_count = sum(1 for a, b in zip(flattened_predictions, flattened_observables) if a != b)

    print(not_matching_count)
    '''