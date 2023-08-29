from pytket.circuit import Circuit
import itertools as it
import numpy as np

def Noise_PauliGadget(strings,t,p):
    Circ = Circuit(len(strings))
    for i in range(len(strings)):
        if strings[i] == 'Z' or strings[i] == 'I':
            continue
        elif strings[i] =='X':
            Circ.H(i)
            add_noise(Circ, p, [i])
        elif strings[i] == 'Y':
            Circ.Sdg(i)
            add_noise(Circ, p, [i])
            Circ.H(i)
            add_noise(Circ, p, [i])
    for j in range(len(strings)-1):
        Circ.CX(j+1,j)
        add_noise(Circ, p, [j+1,j])
    Circ.Rz(t, len(strings)-1)
    for j in range(len(strings)-1,0,-1):
        Circ.CX(j,j-1)
        add_noise(Circ, p, [j,j-1])
    for i in range(len(strings)):
        if strings[i] == 'Z' or strings[i] == 'I':
            continue
        elif strings[i] =='X':
            Circ.H(i)
            add_noise(Circ, p, [i])
        elif strings[i] == 'Y':
            Circ.H(i)
            add_noise(Circ, p, [i])
            Circ.S(i)
            add_noise(Circ, p, [i]) 
    return Circ

def sim_noise(num_qubits,param):
    if not isinstance(num_qubits, int) or num_qubits < 1:
        raise Exception("num_qubits must be a positive integer.")
    # Check that the depolarizing parameter gives a valid CPTP
    num_terms = 4**num_qubits
    max_param = num_terms / (num_terms - 1)
    if param < 0 or param > max_param:
        raise Exception("Depolarizing parameter must be in between 0 " "and {}.".format(max_param))

    # Rescale completely depolarizing channel error probs
    # with the identity component removed
    prob_iden = 1 - param / max_param
    prob_pauli = param / num_terms
    probs = [prob_iden] + (num_terms - 1) * [prob_pauli]
    # Generate pauli strings. The order doesn't matter as long
    # as the all identity string is first.
    paulis = ["".join(tup) for tup in it.product(["I", "X", "Y", "Z"], repeat=num_qubits)]
    return paulis

def add_noise(Circ, p, pos_qubit):
    # if num_qubits != len(pos_qubit):
    #     raise Exception('Need same number of qubits and error gates')
    x = np.random.random()
    paulis = ["".join(tup) for tup in it.product(["I", "X", "Y", "Z"], repeat=len(pos_qubit))][1:]
    if x <= p:
        y = np.random.choice(paulis)
        for i in range(len(pos_qubit)):
            if y[i]=='X':
                Circ.X(pos_qubit[i])
            elif y[i]=='Y':
                Circ.Y(pos_qubit[i])
            elif y[i]=='Z':
                Circ.Z(pos_qubit[i])
    return Circ