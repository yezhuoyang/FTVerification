'''
Surface code array for Z error
Author: A.W
'''

from typing import Callable
import os

def heavy_square_4_circ(gate_noise: float, rounds: int, distance : int = 3) -> str:
    
    w = 4*(distance-1) + 1
    h = 4*(distance-1) + 1

    # generate surface code array
    qbit_dict = {}
    cnt = 0
    for i in range(w):
        for j in range(h):
            qbit_dict[(j, i)], cnt = cnt, cnt + 1
    # print(qbit_dict)
    
    '''Convert 2D coordinate into 1D qubit order'''
    T : Callable[[int, int], int] = lambda x, y: qbit_dict[(x, y)]
    TARGET : Callable[[list], str] = lambda qlist: "".join([' %d' % T(x, y) for x, y in qlist])
    X_ERROR : Callable[[float, list], str] = lambda error_rate, qlist: 'X_ERROR(%lf)' % (error_rate) + TARGET(qlist) # X errors
    R : Callable[[list], str]  = lambda qlist: 'R' + TARGET(qlist) # reset to |0>.
    TICK = 'TICK' # used as a barrier for different layer of circuit
    DEPOLARIZE1 : Callable[[float, list], str] = lambda error_rate, qlist: 'DEPOLARIZE1(%lf)' % (error_rate) + TARGET(qlist) # single-qubit Depolarization errors
    DEPOLARIZE2 : Callable[[float, list], str] = lambda error_rate, qlist: 'DEPOLARIZE2(%lf)' % (error_rate) + TARGET(qlist) # two-qubit Depolarization errors
    H : Callable[[list], str]  = lambda qlist: 'H' + TARGET(qlist) # H gate
    CX : Callable[[list], str]  = lambda qlist: 'CX' + TARGET(qlist) # CX gate
    # MR : Callable[[list], str]  = lambda qlist: 'MR' + TARGET(qlist) # MR gate, measure and reset
    # M : Callable[[list], str]  = lambda qlist: 'M' + TARGET(qlist) # M gate, measure and reset
    LF : Callable[[tuple], tuple] = lambda qbit: (qbit[0], qbit[1]-1)
    RF : Callable[[tuple], tuple] = lambda qbit: (qbit[0], qbit[1]+1)
    UP : Callable[[tuple], tuple] = lambda qbit: (qbit[0]-1, qbit[1])
    DW : Callable[[tuple], tuple] = lambda qbit: (qbit[0]+1, qbit[1])
    
    meas_list = [] # measure list
    def MR(qlist: list, flag: int) -> str:
        '''flag: X stabilizer: 1, Z stabilizer: 2, logical qubit: 3.'''
        _ = [ meas_list.append(qbit+(flag,)) for qbit in qlist ]
        return 'MR' + TARGET(qlist)

    def M(qlist: list, flag: int) -> str:
        '''flag: X stabilizer: 1, Z stabilizer: 2, logical qubit: 3.'''
        _ = [ meas_list.append(qbit+(flag,)) for qbit in qlist ]
        return 'M' + TARGET(qlist)
    M1 = lambda qbit, flag: max(loc for loc, val in enumerate(meas_list) if val == (qbit+(flag,))) - len(meas_list)
    M2 = lambda qbit, flag: max([loc for loc, val in enumerate(meas_list) if val == (qbit+(flag,))][:-1]) - len(meas_list)

    newline = os.linesep
    
    # error_rate = 0.005
    after_reset_flip_error = 0 # error_rate # this will incur error decomposition failing.
    before_measure_flip_error = gate_noise
    before_round_data_depolarization = gate_noise
    after_clifford_depolarization = gate_noise

    # for this model, 
    dqbits = []
    for i in range(0, (w-1)//2 + 1, 2):
        for j in range(-i,i+1,2):
            if i % 4 == j % 4:
                dqbits.append(((h-1)//2-j, i))
    for i in range((w-1)//2 + 2, w, 2):
        for j in range(-(w-1-i),(w-i),2):
            if i % 4 == j % 4:
                dqbits.append(((h-1)//2-j, i))

    mxqbits_four = [] # X syndrome qubit
    mzqbits_four = []
    for i in range(2, (w-1)//2 + 1, 2):
        for j in range(-(i-2),i-2+1,2):
            if (i-2) % 4 == j % 4:
                if i % 4 == 0:
                    mzqbits_four.append(((h-1)//2-j, i))
                else:
                    mxqbits_four.append(((h-1)//2-j, i))
    for i in range((w-1)//2 + 2, w-2, 2):
        for j in range(-(w-3-i),(w-2-i),2):
            if (i-2) % 4 == j % 4:
                if i % 4 == 0:
                    mzqbits_four.append(((h-1)//2-j, i))
                else:
                    mxqbits_four.append(((h-1)//2-j, i))

    mzqbits_two = []
    for i in range(0, (w-1)//2, 4):
        mzqbits_two.append(((h-1)//2-i-2, i))
    for i in range((w-1)//2+4, w, 4):
        mzqbits_two.append((h-1+2-(i-(w-1)//2), i))
    
    mxqbits_two = []
    for i in range(2, (w-1)//2, 4):
        mxqbits_two.append(((h-1)//2+i+2, i))
    for i in range((w-1)//2+2, w, 4):
        mxqbits_two.append(((i - (w-1)//2 -2), i))
    
    mxsignal = []
    for i in mxqbits_four:
        mxsignal.extend([i, DW(i), UP(i), LF(i), RF(i)])
    for i in mxqbits_two:
        if i[1] > (w-1)//2:
            mxsignal.extend([i, LF(i), DW(i)])
        else:
            mxsignal.extend([i, RF(i), UP(i)])

    mzsignal = []
    for i in mzqbits_four:
        mzsignal.extend([i, DW(i), UP(i), LF(i), RF(i)])
    for i in mzqbits_two:
        if i[0] < (h-1)//2:
            mzsignal.extend([i, RF(i), DW(i)])
        else:
            mzsignal.extend([i, LF(i), UP(i)])
    # for ar in [dqbits, mxqbits_four, mzqbits_four, mxqbits_two, mzqbits_two, mxsignal, mzsignal]:
    #     print(ar)

    mqbits = list(set(mxqbits_four + mxqbits_two + mzqbits_four + mzqbits_two))

    allqubits = list(qbit_dict.keys())

    qbit_idx = {}
    for qb, idx in qbit_dict.items():
        qbit_idx[idx] = qb

    circ = ""
    for i in qbit_idx.keys():
        circ += "QUBIT_COORDS(%d, %d) %d\n" % (qbit_idx[i][0], qbit_idx[i][1], i)

    circ += R(allqubits) + newline  # reset data qubits
    circ += X_ERROR(after_reset_flip_error, dqbits) + newline #  after reset flip error on data qubits
    circ += TICK + newline
    circ += DEPOLARIZE1(before_round_data_depolarization, dqbits) + newline

    def all_stabilizer():
        '''Construct all stabilizers'''
        s = ""
        qlist = [qb for qb in mxqbits_four + mxqbits_two] + list(set(mzsignal)-set(mzqbits_two)-set(mzqbits_four))
        s += H(qlist) + newline + DEPOLARIZE1(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mxqbits_four:            
            qlist.extend([i, DW(i)])
        for i in mxqbits_two:
            if i[1] > (w-1)//2:
                qlist.extend([i, DW(i)])
        for i in mzqbits_four:            
            qlist.extend([DW(i), i])
        for i in mzqbits_two:
            if i[0] < (h-1)//2:
                qlist.extend([DW(i), i])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mxqbits_four:            
            qlist.extend([i, LF(i), DW(i), DW(DW(i))])
        for i in mxqbits_two:
            if i[1] > (w-1)//2:
                qlist.extend([i, LF(i), DW(i), DW(DW(i))])
            else:
                qlist.extend([])
        for i in mzqbits_four:            
            qlist.extend([RF(i), i, DW(DW(i)), DW(i)])
        for i in mzqbits_two:
            if i[0] < (h-1)//2:
                qlist.extend([RF(i), i, DW(DW(i)), DW(i)])
            else:
                qlist.extend([])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mxqbits_four:            
            qlist.extend([i, RF(i), LF(i), LF(LF(i))])
        for i in mxqbits_two:
            if i[1] > (w-1)//2:
                qlist.extend([LF(i), LF(LF(i))])
            else:
                qlist.extend([i, RF(i)])
        for i in mzqbits_four:            
            qlist.extend([LF(i), i, RF(RF(i)), RF(i)])
        for i in mzqbits_two:
            if i[0] < (h-1)//2:
                qlist.extend([RF(RF(i)), RF(i)])
            else:
                qlist.extend([LF(i), i])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mxqbits_four:            
            qlist.extend([i, UP(i), RF(i), RF(RF(i))])
        for i in mxqbits_two:
            if i[1] > (w-1)//2:
                qlist.extend([])
            else:
                qlist.extend([i, UP(i), RF(i), RF(RF(i))])
        for i in mzqbits_four:            
            qlist.extend([UP(i), i, LF(LF(i)), LF(i)])
        for i in mzqbits_two:
            if i[0] < (h-1)//2:
                qlist.extend([])
            else:
                qlist.extend([UP(i), i, LF(LF(i)), LF(i)])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mxqbits_four:            
            qlist.extend([i, DW(i), UP(i), UP(UP(i))])
        for i in mxqbits_two:
            if i[1] > (w-1)//2:
                qlist.extend([i, DW(i)])
            else:
                qlist.extend([UP(i), UP(UP(i))])
        for i in mzqbits_four:            
            qlist.extend([DW(i), i, UP(UP(i)), UP(i)])
        for i in mzqbits_two:
            if i[0] < (h-1)//2:
                qlist.extend([DW(i), i])
            else:
                qlist.extend([UP(UP(i)), UP(i)])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mxqbits_four:            
            qlist.extend([i, LF(i)])
        for i in mxqbits_two:
            if i[1] > (w-1)//2:
                qlist.extend([i, LF(i)])
            else:
                qlist.extend([])
        for i in mzqbits_four:            
            qlist.extend([RF(i), i])
        for i in mzqbits_two:
            if i[0] < (h-1)//2:
                qlist.extend([RF(i), i])
            else:
                qlist.extend([])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mxqbits_four:            
            qlist.extend([i, RF(i)])
        for i in mxqbits_two:
            if i[1] > (w-1)//2:
                qlist.extend([])
            else:
                qlist.extend([i, RF(i)])
        for i in mzqbits_four:            
            qlist.extend([LF(i), i])
        for i in mzqbits_two:
            if i[0] < (h-1)//2:
                qlist.extend([])
            else:
                qlist.extend([LF(i), i])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mxqbits_four:            
            qlist.extend([i, UP(i)])
        for i in mxqbits_two:
            if i[1] > (w-1)//2:
                qlist.extend([])
            else:
                qlist.extend([i, UP(i)])
        for i in mzqbits_four:            
            qlist.extend([UP(i), i])
        for i in mzqbits_two:
            if i[0] < (h-1)//2:
                qlist.extend([])
            else:
                qlist.extend([UP(i), i])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        
        qlist = [qb for qb in mxqbits_four + mxqbits_two] + list(set(mzsignal)-set(mzqbits_two)-set(mzqbits_four))
        s += H(qlist) + newline + DEPOLARIZE1(after_clifford_depolarization, qlist) + newline + TICK + newline
        ##################################
        qlist = mxsignal + mzsignal
        MR(mxsignal, 1), MR(mzsignal, 2)
        s += X_ERROR(before_measure_flip_error, qlist) + newline + 'MR' + TARGET(qlist) + newline + X_ERROR(after_reset_flip_error, qlist) + newline
        return s

    def X_stabilizer():
        '''Construct all stabilizers'''
        s = ""
        qlist = [qb for qb in mxqbits_four + mxqbits_two]
        s += H(qlist) + newline + DEPOLARIZE1(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mxqbits_four:            
            qlist.extend([i, DW(i)])
        for i in mxqbits_two:
            if i[1] > (w-1)//2:
                qlist.extend([i, DW(i)])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mxqbits_four:            
            qlist.extend([i, UP(i), DW(i), DW(DW(i))])
        for i in mxqbits_two:
            if i[1] > (w-1)//2:
                qlist.extend([DW(i), DW(DW(i))])
            else:
                qlist.extend([i, UP(i)])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mxqbits_four:            
            qlist.extend([i, LF(i), UP(i), UP(UP(i))])
        for i in mxqbits_two:
            if i[1] > (w-1)//2:
                qlist.extend([i, LF(i)])
            else:
                qlist.extend([UP(i), UP(UP(i))])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mxqbits_four:            
            qlist.extend([i, RF(i), LF(i), LF(LF(i))])
        for i in mxqbits_two:
            if i[1] > (w-1)//2:
                qlist.extend([LF(i), LF(LF(i))])
            else:
                qlist.extend([i, RF(i)])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mxqbits_four:            
            qlist.extend([i, DW(i), RF(i), RF(RF(i))])
        for i in mxqbits_two:
            if i[1] > (w-1)//2:
                qlist.extend([i, DW(i)])
            else:
                qlist.extend([RF(i), RF(RF(i))])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mxqbits_four:            
            qlist.extend([i, UP(i)])
        for i in mxqbits_two:
            if i[1] > (w-1)//2:
                qlist.extend([])
            else:
                qlist.extend([i, UP(i)])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mxqbits_four:            
            qlist.extend([i, LF(i)])
        for i in mxqbits_two:
            if i[1] > (w-1)//2:
                qlist.extend([i, LF(i)])
            else:
                qlist.extend([])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mxqbits_four:            
            qlist.extend([i, RF(i)])
        for i in mxqbits_two:
            if i[1] > (w-1)//2:
                qlist.extend([])
            else:
                qlist.extend([i, RF(i)])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        
        qlist = [qb for qb in mxqbits_four + mxqbits_two]
        s += H(qlist) + newline + DEPOLARIZE1(after_clifford_depolarization, qlist) + newline + TICK + newline
        ##################################
        qlist = mxsignal
        s += X_ERROR(before_measure_flip_error, qlist) + newline + MR(mxsignal, 1) + newline + X_ERROR(after_reset_flip_error, qlist) + newline
        return s

    def Z_stabilizer():
        '''Construct all stabilizers'''
        s = ""
        qlist = list(set(mzsignal)-set(mzqbits_two)-set(mzqbits_four))
        s += H(qlist) + newline + DEPOLARIZE1(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mzqbits_four:            
            qlist.extend([DW(i), i])
        for i in mzqbits_two:
            if i[0] < (h-1)//2:
                qlist.extend([DW(i), i])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mzqbits_four:            
            qlist.extend([UP(i), i, DW(DW(i)), DW(i)])
        for i in mzqbits_two:
            if i[0] < (h-1)//2:
                qlist.extend([DW(DW(i)), DW(i)])
            else:
                qlist.extend([UP(i), i])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mzqbits_four:            
            qlist.extend([LF(i), i, UP(UP(i)), UP(i)])
        for i in mzqbits_two:
            if i[0] < (h-1)//2:
                qlist.extend([])
            else:
                qlist.extend([LF(i), i, UP(UP(i)), UP(i)])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mzqbits_four:            
            qlist.extend([RF(i), i, LF(LF(i)), LF(i)])
        for i in mzqbits_two:
            if i[0] < (h-1)//2:
                qlist.extend([RF(i), i])
            else:
                qlist.extend([LF(LF(i)), LF(i)])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mzqbits_four:            
            qlist.extend([DW(i), i, RF(RF(i)), RF(i)])
        for i in mzqbits_two:
            if i[0] < (h-1)//2:
                qlist.extend([DW(i), i, RF(RF(i)), RF(i)])
            else:
                qlist.extend([])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mzqbits_four:            
            qlist.extend([UP(i), i])
        for i in mzqbits_two:
            if i[0] < (h-1)//2:
                qlist.extend([])
            else:
                qlist.extend([UP(i), i])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mzqbits_four:            
            qlist.extend([LF(i), i])
        for i in mzqbits_two:
            if i[0] < (h-1)//2:
                qlist.extend([])
            else:
                qlist.extend([LF(i), i])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mzqbits_four:            
            qlist.extend([RF(i), i])
        for i in mzqbits_two:
            if i[0] < (h-1)//2:
                qlist.extend([RF(i), i])
            else:
                qlist.extend([])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        
        qlist = list(set(mzsignal)-set(mzqbits_two)-set(mzqbits_four))
        s += H(qlist) + newline + DEPOLARIZE1(after_clifford_depolarization, qlist) + newline + TICK + newline
        ##################################
        qlist = mzsignal
        s += X_ERROR(before_measure_flip_error, qlist) + newline + MR(mzsignal, 2) + newline + X_ERROR(after_reset_flip_error, qlist) + newline
        return s
    
    def X_measure(flag : int = 0) -> str:
        '''flag: 0, initial; 1, compare to before'''
        s = ""
        qlist = mxsignal
        for qb in qlist:
            if flag == 0:
                s += "DETECTOR(%d,%d,0) rec[%d]" % (qb[0],qb[1],M1(qb, 1)) + newline # may cause conflict with Z-reset
            elif flag == 1:
                s += "DETECTOR(%d,%d,0) rec[%d] rec[%d]" % (qb[0],qb[1],M1(qb, 1),M2(qb, 1)) + newline
        return s
    def Z_measure(flag : int = 0) -> str:
        '''flag: 0, initial; 1, compare to before'''
        s = ""
        qlist = mzsignal
        for qb in qlist:
            if flag == 0:
                s += "DETECTOR(%d,%d,0) rec[%d]" % (qb[0],qb[1],M1(qb, 2)) + newline
            elif flag == 1:
                s += "DETECTOR(%d,%d,0) rec[%d] rec[%d]" % (qb[0],qb[1],M1(qb, 2), M2(qb, 2)) + newline
        return s

    # circ += X_stabilizer()
    # circ += X_measure(0)

    # circ += Z_stabilizer()
    circ += all_stabilizer()
    circ += Z_measure(0)
    # Enter the repetition section
    def rep(rounds):
        s = ""
        if rounds > 0:
            s += "REPEAT %d {" % rounds + newline
            s += TICK + newline + DEPOLARIZE1(before_round_data_depolarization, dqbits) + newline
            s += all_stabilizer()
            ''' first assert X measurements preserves '''
            s += "SHIFT_COORDS(%d, %d, %d)" % (0,0,1) + newline
            s += X_measure(1)
            ''' second assert Z measurements preserves '''
            s += "SHIFT_COORDS(%d, %d, %d)" % (0,0,1) + newline
            s += Z_measure(1)
            s += "}" + newline
        return s

    circ += rep(rounds)

    circ += X_ERROR(before_measure_flip_error, dqbits) + newline + M(dqbits, 3) + newline

    def assert_Z_stabilizers():
        ''' let me assert Z stabilizers '''
        s = ""
        for qb in mzqbits_four:
            dqa, dqb, dqc, dqd = DW(DW(qb)), UP(UP(qb)), LF(LF(qb)), RF(RF(qb))
            s += "DETECTOR(%d,%d,0) rec[%d] rec[%d] rec[%d] rec[%d] rec[%d]" % (qb[0],qb[1],M1(dqa, 3), M1(dqb, 3), M1(dqc, 3), M1(dqd, 3), M1(qb, 2)) + newline
        for qb in mzqbits_two:
            if qb[0] < (h-1)//2:
                s += "DETECTOR(%d,%d,0) rec[%d] rec[%d] rec[%d]" % (qb[0],qb[1],M1(DW(DW(qb)), 3), M1(RF(RF(qb)), 3), M1(qb, 2)) + newline
            else:
                s += "DETECTOR(%d,%d,0) rec[%d] rec[%d] rec[%d]" % (qb[0],qb[1],M1(UP(UP(qb)), 3), M1(LF(LF(qb)), 3), M1(qb, 2)) + newline
        return s

    def assert_Z_logical():
        ''' assert the Z logical operation '''
        s = ""
        qlist = []
        for i in range(0, w, 4):
            qlist.append(((h-1)//2, i))
        s += "OBSERVABLE_INCLUDE(0)"
        for qb in qlist:
            s += " rec[%d]" % M1(qb, 3)
        s += newline
        return s

    circ += assert_Z_stabilizers()
    circ += assert_Z_logical()
    
    return circ

if __name__ == '__main__':
    from mwpm import *
    import stim
    import matplotlib.pylab as plt

    rounds = 1
    distance = 3
    circ = heavy_square_4_circ(0.005,rounds, distance)
    print(circ)
    print(repr(stim.Circuit(circ).detector_error_model()))
    


    for run in range(1):
        num_shots = 100
        noise_range = [0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0045, 0.0050, 0.01, 0.015, 0.020, 0.025, 0.030]
        noise_range = [noise for noise in noise_range]
        bname = 'Heavy Square-4' 
        xss = []
        yss = []
        for d in [3,5,7]:
            xs = []
            ys = []
            for noise in noise_range:
                circstr = heavy_square_4_circ(noise, d*3, d)
                circuit = stim.Circuit(circstr)
                xs.append(noise)
                logical_error_rate=count_logical_errors(circuit, num_shots) / num_shots
                print("d=",d," noise=",noise,"logical_error_rate=",logical_error_rate)
                ys.append(logical_error_rate)
            plt.plot(xs, ys, label=bname+" d=" + str(d))
            xss.append(xs)
            yss.append(ys)
        plt.semilogy()
        plt.xlabel("physical error rate")
        plt.ylabel("logical error rate")
        plt.legend()
        plt.savefig(bname+"-run-%d-shots-%d222.pdf" % (run, num_shots))
        np.savez(bname+"-run-%d-shots-%d" % (run, num_shots), xss=xss, yss=yss)
