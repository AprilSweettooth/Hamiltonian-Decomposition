from utils.plot import evol_plot
from utils.func import count_parity
from trotter.hamsimtrotter import AlgorithmHamSimTrotter
from utils.term_grouping import check_commutivity

from pytket.pauli import Pauli
from pytket.circuit import Circuit, PauliExpBox
from pytket.extensions.qiskit import AerBackend, AerStateBackend
from pytket.transform import Transform
from pytket.extensions.qiskit import tk_to_qiskit

import numpy as np
import math as m
import matplotlib.pyplot as plt

class AlgorithmHamSimqDrift:
    """ The QDrift Trotterization method, which selects each each term in the
    Trotterization randomly, with a probability proportional to its weight. Based on the work
    of Earl Campbell in https://arxiv.org/abs/1811.08017.
    """
    def __init__(self,
                 initial_state: Circuit, 
                 qubit_operator: list,
                 coeff: list,
                 t_max: float,
                 reps: int,
                 seg: int,
                 *args,
                 **kwargs):

        self.circuit = initial_state
    
        self._n_qubits = initial_state.n_qubits
        self._initial_state = AerStateBackend().run_circuit(initial_state).get_state()
        self._qubit_operator = qubit_operator
   
        self.coeff = coeff
        self.t_max = t_max
        self.N = seg
        self._rep = reps
        self._time_step = t_max/(self._rep*self.N)
        self._time_space = np.linspace(0,t_max,self._rep+1)

        self.backend = AerBackend()
        self.statebackend = AerStateBackend()
        self.U_sims = []
        self.shots = []
        self.terms = 0
        self.depth = []
        self.p = None
        # self.U_sim = np.zeros((2**self._n_qubits,2**self._n_qubits)) 

    def sampling(self,idx, prob):
        sample_index = []
        probs = []
        while len(sample_index) < self._rep:
            H_j = np.random.choice(idx,p=prob)
            x = np.random.random()
            if len(sample_index) > 1:
                if H_j == sample_index[-1] and H_j == sample_index[-2]:
                    if x < prob[idx.index(H_j)]**3:
                        sample_index.append(H_j)
                        probs.append(prob[idx.index(H_j)]**3)
                elif H_j == sample_index[-1]:
                    if x < prob[idx.index(H_j)]**2:
                        sample_index.append(H_j)
                        probs.append(prob[idx.index(H_j)]**2)
                else:
                    if x < prob[idx.index(H_j)]:
                        sample_index.append(H_j)
                        probs.append(prob[idx.index(H_j)])
            elif len(sample_index) == 1:
                if H_j == sample_index[-1]:
                    if x < prob[idx.index(H_j)]**2:
                        sample_index.append(H_j)
                        probs.append(prob[idx.index(H_j)]**2)
                else:
                    if x < prob[idx.index(H_j)]:
                        sample_index.append(H_j)
                        probs.append(prob[idx.index(H_j)])
            else:
                if x < prob[idx.index(H_j)]:
                    sample_index.append(H_j)
                    probs.append(prob[idx.index(H_j)])        
        return sample_index, probs        


    def Drift_step(self, abs_coeff):
        weights = []
        for i in range(len(self.coeff)):
            if isinstance(self.coeff[i],list):
                if not abs_coeff:
                    weights.append(np.mean([self.coeff[i]]))
                else:
                    weights.append(sum([abs(c) for c in self.coeff[i]]))
            else:
                raise Exception("Sorry, please give as lists of list")
        abs_weights = [abs(w) for w in weights]
        lambd = sum(abs_weights)
        prob = [weight / lambd for weight in abs_weights]
        self.p = prob
        op_index = list(range(0,len(weights)))
        # if not abs_coeff:
        sample_index = np.random.choice(op_index, self.N*self._rep, p=prob)
        # else:
        # sample_index, probs = self.sampling(op_index, prob)
            # print(probs)
        sample_ops = [self._qubit_operator[s] for s in sample_index]
        new_coeff = lambd * (self.t_max/(np.pi/2)) / self._rep
        # if random:
            # new_coeffs = [new_coeff*co/(abs(co)*p) for co,p in zip(weights,probs)]
        # else:
        new_coeffs = [new_coeff*weights[idx]/(abs(weights[idx])) for idx in sample_index]
        # print(new_coeffs,sample_index)
        # print(sample_ops,new_coeffs,sample_index)
        return sample_ops,new_coeffs,sample_index
    
    def Convert_String_to_Op(self,paulis):
        ops = [0]*self._n_qubits
        # print(paulis)
        # if len(paulis) != self._n_qubits:
        #     return [Pauli.I]*self._n_qubits
        for i in range(self._n_qubits):
            if paulis[i] == 'I':
                ops[i] = Pauli.I
            elif paulis[i] == 'X':
                ops[i] = Pauli.X
            elif paulis[i] == 'Y':
                ops[i] = Pauli.Y
            elif paulis[i] == 'Z':
                ops[i] = Pauli.Z
        return ops

    def replacer(self, s, index, newstring):
        return s[:index] + newstring + s[index + 1:]

    def construct_protection(self,n,conj=False):
        protection = Circuit(self._n_qubits) 
        c = Circuit(self._n_qubits)
        for i in range(self._n_qubits):
            c.Ry(0.25,i)
        Cir = c.copy()
        c_dagger = Cir.dagger()
        protection.append(c)
        pbox = PauliExpBox([Pauli.Z]*self._n_qubits, 0.02*n)
        protection.add_pauliexpbox(pbox, np.arange(self._n_qubits)) 
        protection.append(c_dagger)
        if conj:
            return protection.dagger()
        else:
            return protection

    def construct_number_protection(self,n,conj=False):
        # protection = Circuit(self._n_qubits)
        circ = Circuit(self._n_qubits)
        for i in range(self._n_qubits):
            circ.Rz(0.02*n,i)
            # circ.Rz(-0.25,i)
            # circ.Rz(-0.25,(i+1)%self._n_qubits)
            # pbox = PauliExpBox(self.Convert_String_to_Op(self.replacer(self.replacer('I'*self._n_qubits,i,'Z'),(i+1)%self._n_qubits,'Z')), 0.02*n) 
            # circ.add_pauliexpbox(pbox, np.arange(self._n_qubits))
        if conj:
            return circ.dagger()
        else:
            return circ
        
    def trotter(self, order=1, spectral=True, protected=True):
        time_step = self._rep
        for n in range(time_step+1):
            if n==0:
                circ = self.circuit.copy()
                self.depth.append(0)
            else:
                if n==1:
                    circ = self.circuit.copy()
                if order==2:
                    ops = self._qubit_operator + self._qubit_operator[::-1] 
                    co = self.coeff + self.coeff[::-1]
                    if protected:
                        circ.append(self.construct_number_protection(n))
                    for i in range(len(ops)):
                        p = self.Convert_String_to_Op(ops[i])
                        pbox = PauliExpBox(p, co[i]/(time_step*np.pi))
                        circ.add_pauliexpbox(pbox, np.arange(self._n_qubits))  
                    self.depth.append(self.depth[-1]+len(ops))  
                else:
                    ops = self._qubit_operator
                    if protected:
                        circ.append(self.construct_number_protection(n))
                    for i in range(len(ops)):
                        p = self.Convert_String_to_Op(ops[i])
                        pbox = PauliExpBox(p,self.coeff[i]/(time_step*np.pi/2))
                        circ.add_pauliexpbox(pbox, np.arange(self._n_qubits)) 
                    self.depth.append(self.depth[-1]+len(ops))  

            if protected:
                circ.append(self.construct_number_protection(n,True)) 
            naive_circuit = circ.copy()
            Transform.DecomposeBoxes().apply(naive_circuit)
            
            # print(tk_to_qiskit(naive_circuit))
            if spectral:
                self.U_sims.append(naive_circuit.get_unitary())
                
            else:
                naive_circuit.measure_all()
                compiled_circuit = self.backend.get_compiled_circuit(naive_circuit)
                handle = self.backend.process_circuit(compiled_circuit, n_shots=100)
                counts = self.backend.get_result(handle).get_counts()
                self.shots.append(counts)

        if spectral:
            return self.U_sims, self.depth
        else:
            return count_parity(self.shots)          

    def Drift_exp(self, sampled=None, protected=False, spectral=True, abs_coeff=False, trotter=False, depth=1000):
        if sampled != None:
            Vo, coeffo, idxo = sampled[0], sampled[1], sampled[2]
        else:
            Vo, coeffo, idxo = self.Drift_step(abs_coeff)
        
        # print(Vo)
        for v in Vo:
            self.terms += len(v)
        if self.terms < depth:
            V, coeff, idx = Vo, coeffo, idxo 
        else:
            V = []
            coeff = []
            idx = []
            count = 0
            depth_itr = depth
            while depth_itr > 0:
                V.append(Vo[count])
                depth_itr -= len(Vo[count])
                coeff.append(coeffo[count])
                idx.append(idxo[count])
                count+=1
                # print(V)
            self._rep = depth
        # print(V, coeff, idx)
        # op = [0]*self._n_qubits
        # for i in range(self._n_qubits):
        #     op[i] = Pauli.Z 
        # print(V)
        for n in range(0,min(count, self._rep+1)):
            if n ==0:
                circ = self.circuit.copy()
                self.depth.append(0)
            else:
                if n == 1:
                    circ = self.circuit.copy() 
                # for _ in range(n):
                    # print(V, coeff, idx)
                if protected:
                    # for i in range(self._n_qubits):
                    #     circ.Ry(0.25, i)
                    # pbox = PauliExpBox(identity, 0.02*n)
                    # circ.add_pauliexpbox(pbox, np.arange(self._n_qubits))
                    # for i in range(self._n_qubits):
                    #     circ.Ry(-0.25, i)
                
                # n -= 1
            # cir.append(self.construct_protection(n).dagger())
            # pbox = PauliExpBox([Pauli.Z]*self._n_qubits, -0.02*n) 
            # cir.add_pauliexpbox(pbox, np.arange(self._n_qubits)) 
                    # protect_cir = self.construct_protection(n)
                    # circ.append(protect_cir)
                    # protect_cir_dagger = protect_cir.dagger()
                    circ.append(self.construct_number_protection(n))
                for j in range(self.N):
                    # print((n-1)*self.N+j)
                    if trotter and len(V[(n-1)*self.N+j])>1 and not check_commutivity(V[(n-1)*self.N+j]):
                        # print(V[(n-1)*self.N+j])
                        V1 = V[(n-1)*self.N+j] 
                        V2 = V[(n-1)*self.N+j][::-1]
                        V_new = V1 + V2
                        for i in range(len(V_new)):
                            p = self.Convert_String_to_Op(V_new[i])
                            pbox = PauliExpBox(p, coeff[(n-1)*self.N+j]/2)
                            circ.add_pauliexpbox(pbox, np.arange(self._n_qubits)) 
                        self.depth.append(self.depth[-1]+len(V_new)) 
                    else:
                        # print(n)
                        # print(n,V[(n-1)*self.N+j])
                        for paulis in V[(n-1)*self.N+j]:
                            p = self.Convert_String_to_Op(paulis)
                            pbox = PauliExpBox(p, coeff[(n-1)*self.N+j])
                            circ.add_pauliexpbox(pbox, np.arange(self._n_qubits))
                        self.depth.append(self.depth[-1]+len(V[(n-1)*self.N+j])) 

                if protected:
                    circ.append(self.construct_number_protection(n,True))
                        # circ.append(protect_cir_dagger)
                    #     # for i in range(self._n_qubits):
                    #     for i in range(self._n_qubits):
                    #         circ.Ry(-0.25, i)
                    #     pbox = PauliExpBox(identity, -0.02*n)
                    #     circ.add_pauliexpbox(pbox, np.arange(self._n_qubits))
                    #     for i in range(self._n_qubits):
                    #         circ.Ry(0.25, i)
                            # pbox = PauliExpBox(self.Convert_String_to_Op(self.replacer(identity,i,'Z')), -0.02*n) 
                            # circ.add_pauliexpbox(pbox, np.arange(self._n_qubits))

            naive_circuit = circ.copy()
            Transform.DecomposeBoxes().apply(naive_circuit)
            
            # print(tk_to_qiskit(naive_circuit))
            if spectral:
                self.U_sims.append(naive_circuit.get_unitary())
                
            else:
                naive_circuit.measure_all()
                compiled_circuit = self.backend.get_compiled_circuit(naive_circuit)
                handle = self.backend.process_circuit(compiled_circuit, n_shots=100)
                counts = self.backend.get_result(handle).get_counts()
                self.shots.append(counts)

            # statevec = self.statebackend.run_circuit(compiled_circuit).get_state()
            # self.exp.append(abs(np.vdot(self._initial_state,statevec))**2)
            # self.E[self._time_space[n]] = self.H.state_expectation(statevec, [Qubit(i) for i in range(self._n_qubits)]) 
        if spectral:
            return self.U_sims, [V, coeff, idx], self.depth
        else:
            return count_parity(self.shots), [V, coeff, idx]


    def execute(self, real=None, color='purple', plot=False):
        # trotter_cheat = AlgorithmHamSimTrotter(self.c_copy,self.real_op,self.m,self.t_max,self._rep,fresh_symbol("t"))
        # trotter_cheat._trotter_step_cheat(exps='proj')
        # plt.plot(self._time_space, list(trotter_cheat._real_measurement.values()), c='blue')
        if plot:
            plt.plot(self._time_space, real)
            plt.plot(self._time_space, list(self.E.values()), c=color)
            plt.xlabel(r'Time')
            plt.ylabel('proj')
        else:
            return self.U_sim
        # evol_plot(self._time_space, self.exp, self.gate_count)