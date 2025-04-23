from QEPG import *



def test():
    distance=5
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.001)
    stim_circuit=stim.Circuit.generated("repetition_code:memory",rounds=3*distance,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit) 



    sampler=WSampler(circuit)
    sampler.construct_QPEG()
    sampler.set_shots(100)

    Wlist=[1,2,3,4,5,6,7]

    
    sampler.calc_binomial_weight()

    final_lr=0

    for w in Wlist:

        logical_error_rate=sampler.calc_logical_error_rate_with_fixed_w(10000,w)
        s2=sampler._binomial_weights[w]

        print("WSampler logical error rate: ",logical_error_rate)

        print("Logical contribution:",s2*logical_error_rate)

        final_lr+=s2*logical_error_rate


    print("Final logical error rate: ",final_lr)


if __name__ == "__main__":
    test()