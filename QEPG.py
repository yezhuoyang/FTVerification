from paulitracer import *
from multiprocessing import Process, Pool
import os
import signal
from multiprocessing import shared_memory
from noise_sampler import cython_sample_noise_and_calc_result, sample_fixed_one_two_three
from scipy.optimize import curve_fit


'''
Class of quantum error propagation graph
TODO: Optimize the algorithm to construct the QEPG
'''
class QEPG:

    def __init__(self,circuit:CliffordCircuit):
        self._circuit = circuit 
        self._tracer=PauliTracer(circuit)

        self._total_meas=self._circuit._totalMeas
        self._total_noise=self._circuit._totalnoise
        self._XerrorMatrix=np.zeros((self._total_meas,3*self._total_noise), dtype='uint8')
        self._YerrorMatrix=np.zeros((self._total_meas,3*self._total_noise), dtype='uint8')
        self._ZerrorMatrix=np.zeros((self._total_meas,3*self._total_noise), dtype='uint8')


        
        #print("QEPG total noise:{} ".format(self._total_noise))


    def backword_graph_construction(self):
        nqubit=self._circuit._qubit_num
        #Keep track of the effect of X,Y,Z back propagation
        current_x_prop=np.zeros((nqubit,self._total_meas), dtype='uint8')
        current_y_prop=np.zeros((nqubit,self._total_meas), dtype='uint8')
        current_z_prop=np.zeros((nqubit,self._total_meas), dtype='uint8')
        current_noise_index=self._circuit._totalnoise-1
        current_meas_index=self._total_meas-1  
        total_noise=self._total_noise
        self._detectorMatrix=np.zeros((self._total_meas,3*self._total_noise), dtype='uint8') 
        T=len(self._circuit._gatelists)
        for t in range(T-1,-1,-1):
            #Update current_x_prop, current_y_prop, current_z_prop based on the current gate and measurement
            gate=self._circuit._gatelists[t]
            '''
            If the gate is a oiginal noise, add edges to the graph based on current propogation
            '''
            if isinstance(gate, pauliNoise):
                noiseindex=current_noise_index # TODO: Determine the index of the noise
                #print("Noise!")
                for j in range(self._total_meas):
                    self._detectorMatrix[j][noiseindex]=current_x_prop[gate._qubitindex][j]
                    self._detectorMatrix[j][total_noise+noiseindex]=current_y_prop[gate._qubitindex][j] 
                    self._detectorMatrix[j][total_noise*2+noiseindex]=current_z_prop[gate._qubitindex][j]  
                current_noise_index-=1
                continue
            '''
            When there is a measurement, update the current propogation based on the measurement
            We just need to consider the propagation of X and Y because only 
            the X and Y error can be detected by the measurement
            '''
            if isinstance(gate, Measurement):
                measureindex=current_meas_index # TODO: Determine the index of the noise
                current_x_prop[gate._qubitindex][measureindex]=1
                current_y_prop[gate._qubitindex][measureindex]=1
                current_meas_index-=1
                continue

            if isinstance(gate,Reset):
                for j in range(self._total_meas): 
                    current_x_prop[gate._qubitindex][j]=0
                    current_y_prop[gate._qubitindex][j]=0
                    current_z_prop[gate._qubitindex][j]=0
                continue

            '''
            Deal with propagation by CNOT gate, we need to consider the propagation of X and Z
            '''
            if gate._name=="CNOT":
                control=gate._control
                target=gate._target
                current_x_prop[control,:]=(current_x_prop[control,:]+current_x_prop[target,:])%2
                current_z_prop[target,:]=(current_z_prop[control,:]+current_z_prop[target,:])%2                
                current_y_prop[control,:]=(current_y_prop[control,:]+current_x_prop[target,:])%2
                current_y_prop[target,:]=(current_y_prop[target,:]+current_z_prop[control,:])%2
                continue
            
            '''
            Deal with propagation by H gate
            If there is a H gate, we need to swap the X and Z propagations
            '''
            if gate._name=="H":
                qubitindex=gate._qubitindex
                tmp_row=current_x_prop[qubitindex,:].copy()
                current_x_prop[qubitindex,:]=current_z_prop[qubitindex,:]
                current_z_prop[qubitindex,:]=tmp_row               
                continue




    def compute_graph(self):
        for i in range(self._total_noise):
            #print(f"Calc noise {i}, set it X error")
            self._tracer.reset()
            self._tracer.set_noise_type(i,1)
            self._tracer.prop_all()
            measured_error=self._tracer.get_measuredError()
            #print(f"Measured error: {measured_error}")
            for j in range(self._total_meas):
                if measured_error[j]=='X':
                    #print(f"Add x type edge: {1},{i},{j}")
                    self.add_x_type_edge(1,i,j)
                elif measured_error[j]=='Y':
                    self.add_y_type_edge(1,i,j)
                elif measured_error[j]=='Z':
                    self.add_z_type_edge(1,i,j)
            self._tracer.reset()    
            #print(f"Calc noise {i}, set it Y error")
            self._tracer.set_noise_type(i,2)
            self._tracer.prop_all()
            measured_error=self._tracer.get_measuredError()
            #print(f"Measured error: {measured_error}")
            for j in range(self._total_meas):
                if measured_error[j]=='X':
                    #print(f"Add x type edge: {2},{i},{j}")
                    self.add_x_type_edge(2,i,j)
                elif measured_error[j]=='Y':
                    self.add_y_type_edge(2,i,j)
                elif measured_error[j]=='Z':
                    self.add_z_type_edge(2,i,j)
            self._tracer.reset()    
            #print(f"Calc noise {i}, set it Z error")
            self._tracer.set_noise_type(i,3)
            self._tracer.prop_all()
            measured_error=self._tracer.get_measuredError()
            #print(f"Measured error: {measured_error}")
            for j in range(self._total_meas):
                if measured_error[j]=='X':
                    #print(f"Add x type edge: {3},{i},{j}")
                    self.add_x_type_edge(3,i,j)
                elif measured_error[j]=='Y':
                    self.add_y_type_edge(3,i,j)
                elif measured_error[j]=='Z':
                    self.add_z_type_edge(3,i,j)
        #print("QEPG total noise after compute graph:{} ".format(self._total_noise))
        #print(self._XerrorMatrix.shape)

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


def python_sample_fixed_one_two_three(N, k):
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
        variance=self._logical_error_rate*(1-self._logical_error_rate)/self._shots

        return self._logical_error_rate,variance


    

#Sample noise with weight K
def python_sample_noise_and_calc_result(shots,totalnoise,W,dtype,shm_detec_name,parity_group_length):
    shm_detec = shared_memory.SharedMemory(name=shm_detec_name)
    detectorMatrix = np.ndarray((parity_group_length+1,3*totalnoise), dtype=dtype, buffer=shm_detec.buf)  
    result=[]
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
        #print(dectectorMatrix.shape, noise_vector.shape)
        detectorresult=np.matmul(detectorMatrix, noise_vector)%2
        result.append(detectorresult)
        detection_events.append(list(detectorresult[:parity_group_length]))
        observable_flips.append(list(detectorresult[parity_group_length:]))
    return detection_events,observable_flips



from scipy.stats import norm

def init_worker():
    # Ignore SIGINT in worker processes so that only the main process catches it.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def model_function(x, alpha):
    mu, sigma = 10, 1
    cdf_values = 0.5*norm.cdf(x, loc=mu, scale=alpha)
    return cdf_values


class WSampler():
    def __init__(self, circuit:CliffordCircuit,distance=3):
        self._inducedNoise=["I"]*circuit._qubit_num
        self._measuredError={}
        self._qubit_num=circuit._qubit_num     
        self._circuit=circuit
        self._totalnoise=circuit.get_totalnoise()


        self._sample_nums=[0]*self._totalnoise
        self._sample_sums=[0]*self._totalnoise

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

        #self._binomial_weights=[0]*self._totalnoise
        #self.calc_binomial_weight()


        self._maxvariance=1e-8

        self._stim_str=circuit._stim_str



    def construct_QPEG(self):
        self._QPEGraph=QEPG(self._circuit)
        #self._QPEGraph.compute_graph()
        self._QPEGraph.backword_graph_construction()    


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
        

    
    def calc_logical_error_rate_parallel(self,error_rate_list):


        exp_noise=int(self._totalnoise*self._circuit._error_rate)
        #min_W=max(0,exp_noise-20)
        #max_W=min(self._totalnoise,exp_noise+20)
        min_W=0
        max_W=150
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
        #self._circuit.compile_from_stim_circuit_str(self._circuit._stim_str)
        #self._circuit=CliffordCircuit(2)
        self._circuit.set_error_rate(0.1)
        self._circuit.compile_from_stim_circuit_str(self._circuit._stim_str)    
        self._totalnoise=self._circuit.get_totalnoise()

        #print(f"Total noise we get in main function: {self._totalnoise}")


        total_noise=self._totalnoise
        parity_group=self._circuit.get_parityMatchGroup()
        observable=self._circuit.get_observable()
        QEPGgraph=self._QPEGraph
        total_meas=self._circuit._totalMeas


        detectorMatrix=QEPGgraph._detectorMatrix
        paritymatrix=np.zeros((len(parity_group)+1,total_meas), dtype='uint8')
        for i in range(len(parity_group)):
            for j in parity_group[i]:
                paritymatrix[i][j]=1
        #print("observable: {}".format(observable))
        for i in range(len(observable)):
            paritymatrix[len(parity_group)][observable[i]]=1
        detectorMatrix=np.matmul(paritymatrix,detectorMatrix)
        


        #print("The shape of detector matrix that we construct:")
        #print(detectorMatrix.shape)
        stim_str=self._circuit._stim_str


        shm_dec = shared_memory.SharedMemory(create=True, size=detectorMatrix.nbytes)
        # Create a NumPy array backed by the shared memory
        shared_array = np.ndarray(detectorMatrix.shape, dtype=detectorMatrix.dtype, buffer=shm_dec.buf)
        # Copy the data into shared memory
        shared_array[:] = detectorMatrix[:]    


        parity_group_length=len(parity_group)
        inputs=[]
        for i in range(max_W-min_W+1):
            inputs=inputs+[(self._shots,total_noise,i,detectorMatrix.dtype.name,shm_dec.name, parity_group_length)]
        
        pool = Pool(processes=os.cpu_count(), initializer=init_worker)
        try:
            # starmap is a blocking call that collects results from each process.
            results = pool.starmap(python_sample_noise_and_calc_result, inputs)
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


        result_list=[]
        for error_rate in error_rate_list:

            #self._circuit.set_error_rate(error_rate)
            #self._circuit.compile_from_stim_circuit_str(self._circuit._stim_str)
            self._circuit=CliffordCircuit(2)
            self._circuit.set_error_rate(error_rate)
            self._circuit.compile_from_stim_circuit_str(stim_str)
            
            self._totalnoise=self._circuit.get_totalnoise()

            detector_error_model = self._circuit._stimcircuit.detector_error_model(decompose_errors=True)
            matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
            #print("-------------length of parity match group------------------------------")            
            #print(len(self._circuit._parityMatchGroup))
            predictions = matcher.decode_batch(detections)

            flattened_predictions = [item for sublist in predictions for item in sublist]
            flattened_observables = [item for sublist in observables for item in sublist]   

            self._logical_error_distribution=[0]*self._totalnoise
            self._binomial_weights=[0]*self._totalnoise

            self.calc_binomial_weight()

            #print(self._binomial_weights)
            self._logical_error_rate=0

            for i in range(0,max_W-min_W+1):
                tmp_flattened_predictions= flattened_predictions[i*self._shots:(i+1)*self._shots]
                tmp_flattened_observables= flattened_observables[i*self._shots:(i+1)*self._shots]
                errorshots = sum(1 for a, b in zip(tmp_flattened_predictions, tmp_flattened_observables) if a != b)
                #print("Error shots:")
                #print(errorshots)
                #print(len(self._logical_error_distribution))
                #print(min_W+i)
                self._logical_error_distribution[min_W+i]=errorshots/self._shots   
                self._logical_error_rate+=self._binomial_weights[min_W+i]*self._logical_error_distribution[min_W+i]

            result_list.append(self._logical_error_rate)
        
        #print(f"------------Error distribution--------Total Number error: {self._totalnoise}-------------")
        #print(self._logical_error_distribution)
        #print("------------Binomial weights---------------------")
        #print(self._binomial_weights)
        return result_list



    def calc_sample_sum(self,min_W,max_W):
        self._sample_sums[0]=self._sample_nums[0]
        for i in range(1,max_W-min_W+1):
            self._sample_sums[i]=self._sample_sums[i-1]+self._sample_nums[i]


    def calc_sample_num(self,sampleBudget,min_W,max_W):
        for i in range(min_W,max_W+1):
            self._sample_nums[i]=int(sampleBudget*self._binomial_weights[i])

    '''
    Fit the distribution by 1/2-e^{alpha/W}
    '''
    def fit_curve(self,wlist):
        # Initial guess for alpha
        initial_guess = [1.0]

        # Set bounds: alpha > 0 means the lower bound for alpha is 0
        # You can also set an upper bound if you want, e.g. np.inf for no upper limit
        bounds = (0, np.inf)

        # Perform the curve fit with the bounds
        popt, pcov = curve_fit(
            model_function, 
            wlist, 
            [self._logical_error_distribution[x] for x in wlist], 
            p0=initial_guess, 
            bounds=bounds
        )

        # Extract the best-fit parameter (alpha)
        alpha_fit = popt[0]
        print(f"Fitted alpha: {alpha_fit}")
        return alpha_fit


    def calc_logical_error_rate_by_curve_fitting(self,alpha):
        self._logical_error_rate=0
        self._binomial_weights=[0]*self._totalnoise
        self.calc_binomial_weight()
        for i in range(1,self._totalnoise):
            self._logical_error_rate+=model_function(i,alpha)*self._binomial_weights[i]
        return self._logical_error_rate



    def calc_logical_error_distribution(self,wlist=None):

        shots=self._shots
        total_noise=self._totalnoise
        parity_group=self._circuit.get_parityMatchGroup()
        observable=self._circuit.get_observable()
        QEPGgraph=self._QPEGraph
        total_meas=self._circuit._totalMeas

        detectorMatrix=QEPGgraph._detectorMatrix
        paritymatrix=np.zeros((len(parity_group)+1,total_meas), dtype='uint8')

        #print(len(parity_group))
        for i in range(len(parity_group)):
            for j in parity_group[i]:
                paritymatrix[i][j]=1
        #print("observable: {}".format(observable))
        for i in range(len(observable)):
            paritymatrix[len(parity_group)][observable[i]]=1
        detectorMatrix=np.matmul(paritymatrix,detectorMatrix)
        

        shm_dec = shared_memory.SharedMemory(create=True, size=detectorMatrix.nbytes)
        # Create a NumPy array backed by the shared memory
        shared_array = np.ndarray(detectorMatrix.shape, dtype=detectorMatrix.dtype, buffer=shm_dec.buf)
        # Copy the data into shared memory
        shared_array[:] = detectorMatrix[:]    


        parity_group_length=len(parity_group)

        inputs=[]
        L=total_noise
        if wlist is None:
            wlist=[i for i in range(total_noise)]
        else:
            L=len(wlist)
        for i in range(L):
            inputs=inputs+[(shots,total_noise,wlist[i],detectorMatrix.dtype.name,shm_dec.name, parity_group_length)]

        
        pool = Pool(processes=os.cpu_count(), initializer=init_worker)

        try:
            # starmap is a blocking call that collects results from each process.
            results = pool.starmap(python_sample_noise_and_calc_result, inputs)
        except KeyboardInterrupt:
            # Handle Ctrl-C gracefully.
            print("KeyboardInterrupt received. Terminating pool...")
            pool.terminate()
            pool.join()
            return  # or re-raise if you want to propagate the exception

        self._logical_error_distribution=[0]*self._totalnoise

        self._logical_error_rate=0



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


        for i in range(0,L):
            tmp_flattened_predictions= flattened_predictions[i*self._shots:(i+1)*self._shots]
            tmp_flattened_observables= flattened_observables[i*self._shots:(i+1)*self._shots]
            errorshots = sum(1 for a, b in zip(tmp_flattened_predictions, tmp_flattened_observables) if a != b)
            self._logical_error_distribution[wlist[i]]=errorshots/self._shots
            #self._logical_error_rate+=self._binomial_weights[i]*self._logical_error_distribution[i]

        return self._logical_error_distribution


    def calc_logical_error_rate(self):


        exp_noise=int(self._totalnoise*self._circuit._error_rate)

        wrange=20
        min_W=0
        max_W=80
        self._binomial_weights=[0]*self._totalnoise
        self.calc_binomial_weight()
        self.calc_sample_num(self._shots,min_W,max_W)
        self.calc_sample_sum(min_W,max_W)

        #print(self._sample_nums)
        #print(self._sample_sums)

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

        detectorMatrix=QEPGgraph._detectorMatrix
        paritymatrix=np.zeros((len(parity_group)+1,total_meas), dtype='uint8')

        #print(len(parity_group))
        for i in range(len(parity_group)):
            for j in parity_group[i]:
                paritymatrix[i][j]=1
        #print("observable: {}".format(observable))
        for i in range(len(observable)):
            paritymatrix[len(parity_group)][observable[i]]=1
        detectorMatrix=np.matmul(paritymatrix,detectorMatrix)
        

        shm_dec = shared_memory.SharedMemory(create=True, size=detectorMatrix.nbytes)
        # Create a NumPy array backed by the shared memory
        shared_array = np.ndarray(detectorMatrix.shape, dtype=detectorMatrix.dtype, buffer=shm_dec.buf)
        # Copy the data into shared memory
        shared_array[:] = detectorMatrix[:]    



        parity_group_length=len(parity_group)
        inputs=[]
        for i in range(max_W-min_W+1):
            inputs=inputs+[(self._sample_nums[i],total_noise,i,detectorMatrix.dtype.name,shm_dec.name, parity_group_length)]

        
        pool = Pool(processes=os.cpu_count(), initializer=init_worker)

        try:
            # starmap is a blocking call that collects results from each process.
            results = pool.starmap(python_sample_noise_and_calc_result, inputs)
        except KeyboardInterrupt:
            # Handle Ctrl-C gracefully.
            print("KeyboardInterrupt received. Terminating pool...")
            pool.terminate()
            pool.join()
            return  # or re-raise if you want to propagate the exception

        self._logical_error_distribution=[0]*self._totalnoise

        self._logical_error_rate=0



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

        if self._sample_nums[min_W]>0:
            tmp_flattened_predictions= flattened_predictions[0:self._sample_sums[0]]
            tmp_flattened_observables= flattened_observables[0:self._sample_sums[0]]
            errorshots = sum(1 for a, b in zip(tmp_flattened_predictions, tmp_flattened_observables) if a != b)
            self._logical_error_distribution[min_W]=errorshots/self._sample_nums[min_W]
            self._logical_error_rate+=self._binomial_weights[min_W]*self._logical_error_distribution[min_W]


        for i in range(1,max_W-min_W+1):
            if self._sample_nums[min_W+i]>0: 
                tmp_flattened_predictions= flattened_predictions[self._sample_sums[i-1]:self._sample_sums[i]]
                tmp_flattened_observables= flattened_observables[self._sample_sums[i-1]:self._sample_sums[i]]
                errorshots = sum(1 for a, b in zip(tmp_flattened_predictions, tmp_flattened_observables) if a != b)
                self._logical_error_distribution[min_W+i]=errorshots/self._sample_nums[min_W+i] 
                self._logical_error_rate+=self._binomial_weights[min_W+i]*self._logical_error_distribution[min_W+i]

        for i in range(max_W+1,self._totalnoise):
            self._logical_error_rate+=self._binomial_weights[i]*0.5


        variance=self._logical_error_rate*(1-self._logical_error_rate)/self._shots
        
        #print(f"------------Error distribution--------Total Number error: {self._totalnoise}-------------")
        #print(self._logical_error_distribution)
        #print("------------Binomial weights---------------------")
        #print(self._binomial_weights)
        return self._logical_error_rate,variance




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