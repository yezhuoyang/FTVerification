from QEPG import *




def QEPG_sample():
    distance=3
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.01)
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit) 


    sampler=WSampler(circuit)


    sampler.construct_QPEG()


    shot_list=[10,20,30,40,50,80,100,150,200,300,400,500,600,700,800,900,1000]
    for shot in shot_list:
        sampler.set_shots(shot)
        errorrate,variance=sampler.calc_logical_error_rate()
        print(errorrate,variance)



def stim_sample():
    distance=3
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.01)
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit)           
        
    Nsampler=NaiveSampler(circuit)
    
    shot_list=[10,20,30,40,50,80,100]
    for shot in shot_list:
        Nsampler.set_shots(shot)
        #Nsampler.construct_QPEG()
        errorrate,variance=Nsampler.calc_logical_error_rate()
        print(errorrate,variance)


import matplotlib.pyplot as plt


def compare_two_method():
    distance=3
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.0001)
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit) 


    sampler=WSampler(circuit)
    Nsampler=NaiveSampler(circuit)

    sampler.construct_QPEG()


    shot_list=[1000,2000,3000,4000,5000,8000]
    error_rate_list1=[]
    var_list1=[]
    error_rate_list2=[]
    var_list2=[]
    for shot in shot_list:
        sampler.set_shots(shot)
        errorrate,variance=sampler.calc_logical_error_rate()
        print(errorrate,variance)
        error_rate_list1.append(errorrate)
        var_list1.append(variance)  



        Nsampler.set_shots(shot)
        #Nsampler.construct_QPEG()
        errorrate,variance=Nsampler.calc_logical_error_rate()
        print(errorrate,variance)
        error_rate_list2.append(errorrate)
        var_list2.append(variance)

    # --- Plotting section ---
    plt.figure()

    # Use standard deviation for the error bars (sqrt of variance).
    # If you truly want to use the raw variance, replace `np.sqrt(var_listX)` with `var_listX`.
    plt.errorbar(shot_list, error_rate_list1,
                 yerr=np.sqrt(var_list1),
                 fmt='o-', 
                 label='WSampler')

    plt.errorbar(shot_list, error_rate_list2,
                 yerr=np.sqrt(var_list2),
                 fmt='o-', 
                 label='NaiveSampler')

    plt.xlabel("Shots")
    plt.ylabel("Logical Error Rate")
    plt.title("Comparison of WSampler vs. NaiveSampler")
    plt.legend()

    # Save to PNG file
    plt.savefig("compare_two_method.png")
    plt.close()    







if __name__ == "__main__":

    compare_two_method()