from utils.plot import evol_plot
from utils.func import count_parity
from utils.term_grouping import check_commutivity
from utils.evol_real import get_Hmatrix, pauli_string_to_matrix, U_drift, U_trotter

from trotter.hamsimtrotter import AlgorithmHamSimTrotter
from Pauli_Gadgets.paulis import Noise_PauliGadget

from pytket.pauli import Pauli
from pytket.circuit import Circuit, PauliExpBox
from pytket.extensions.qiskit import AerBackend, AerStateBackend
from pytket.transform import Transform
from pytket.extensions.qiskit import tk_to_qiskit

import numpy as np
import math as m
import matplotlib.pyplot as plt
import functools as ft

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
                 noise_param=0.001,
                 noise=False,
                 M=1,
                 H_matrix=None):

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
        self.U_sims_trotter = []
        self.U_sims = [[],[],[]]
        self.shots = []
        self.terms = 0
        self.depth = []
        self.p = None
        self.Z = sum([ft.reduce(np.kron,[np.array([[1, 0], [0, 1]])]*self._n_qubits) for _ in range(self._n_qubits)]) - sum([pauli_string_to_matrix(self.replacer('I'*self._n_qubits, idx, 'Z')) for idx in range(self._n_qubits)])
        # self.Z = ft.reduce(np.kron,[np.array([[1, 0], [0, -1]])]*self._n_qubits)
        self.H = H_matrix
        self.E = []
        self.M = M
        self.noise = noise
        self.noise_param = noise_param
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
        if not self.noise:
            pbox = PauliExpBox([Pauli.Z]*self._n_qubits, 0.02*n)
            protection.add_pauliexpbox(pbox, np.arange(self._n_qubits)) 
        else:
            protection.append(Noise_PauliGadget('Z'*self._n_qubits,0.02*n,self.noise_param))
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
        
    def trotter(self, order=1, spectral=True, protected=False, measurement=None,rand=False, cheat=False):
        time_step = self._rep
        if cheat:
            if self.noise:
                evol, depthtr = U_trotter(self.circuit.get_unitary(), n=time_step, V=self._qubit_operator, coeff=self.coeff, t_max=self.t_max, n_qubits=self._n_qubits, state=self._initial_state, order=order, protect=protected, rand=rand, measurement=measurement,error=self.noise_param)
            else:
                evol, depthtr = U_trotter(self.circuit.get_unitary(), n=time_step, V=self._qubit_operator, coeff=self.coeff, t_max=self.t_max, n_qubits=self._n_qubits, state=self._initial_state, order=order, protect=protected, rand=rand, measurement=measurement)
        else:
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
                            # p = self.Convert_String_to_Op(ops[i])
                            if not self.noise:
                                # pbox = PauliExpBox(p, self.t_max*co[i]/(time_step*np.pi))
                                # circ.add_pauliexpbox(pbox, np.arange(self._n_qubits)) 
                                circ.append(Noise_PauliGadget(ops[i],self.t_max*co[i]/(time_step*np.pi),0)) 
                            else:
                                circ.append(Noise_PauliGadget(ops[i],self.t_max*co[i]/(time_step*np.pi),self.noise_param))
                        self.depth.append(self.depth[-1]+len(ops))  
                    else:
                        if rand:
                            ops = np.random.permutation(self._qubit_operator)
                            # print(ops)
                        else:
                            ops = self._qubit_operator
                        if protected:
                            circ.append(self.construct_number_protection(n))
                        for i in range(len(ops)):
                            # p = self.Convert_String_to_Op(ops[i])
                            if not self.noise:
                                # pbox = PauliExpBox(p,self.t_max*self.coeff[i]/(time_step*np.pi/2))
                                # circ.add_pauliexpbox(pbox, np.arange(self._n_qubits))
                                # print(ops[i])
                                circ.append(Noise_PauliGadget(ops[i],self.t_max*self.coeff[i]/(time_step*np.pi/2),0))
                                # print(i)
                                # print(tk_to_qiskit(circ))
                                # print(circ.depth(), circ.n_gates) 
                            else:
                                circ.append(Noise_PauliGadget(ops[i],self.t_max*self.coeff[i]/(time_step*np.pi/2),self.noise_param)) 
                        self.depth.append(self.depth[-1]+len(ops)) 
                    

                if protected:
                    circ.append(self.construct_number_protection(n,True)) 
                naive_circuit = circ.copy()
            # Transform.DecomposeBoxes().apply(naive_circuit)
            
            # print(tk_to_qiskit(naive_circuit))
            if spectral:
                self.U_sims_trotter.append(naive_circuit.get_unitary())
                
            else:
                # naive_circuit.measure_all()
                # compiled_circuit = self.backend.get_compiled_circuit(naive_circuit)
                # handle = self.backend.process_circuit(compiled_circuit, n_shots=100)
                # counts = self.backend.get_result(handle).get_counts()
                # self.shots.append(counts)
                statevec = naive_circuit.get_statevector()
                if measurement=='H':
                    self.E.append((statevec.conj().T@self.H)@statevec)
                elif measurement=='Z':
                    self.E.append((statevec.conj().T@self.Z)@statevec) 

        if spectral:
            if cheat:
                return evol, depthtr
            else:
                return self.U_sims_trotter, self.depth
        else:
            if cheat:
                return evol, depthtr
            else:
                return self.E, self.depth          

    def Drift_exp(self, sampled=None, protected=False, spectral=True, abs_coeff=False, trotter=False, depth=1000, measurement=None,cheat=False):
        if self.M != 1:
            self.depth = [[],[],[]]
            Vs, coeffs, idxs = [],[],[] 
            evols = [[],[],[]]
        else:
            Vs, coeffs, idxs = [],[],[] 
            self.depth = [[]]
            evols = [[]]
        for m in range(self.M):
            if sampled == None:
                Vo, coeffo, idxo = self.Drift_step(abs_coeff)
                
                # print(Vo)
                for v in Vo:
                    self.terms += len(v)
                # if self.terms < depth:
                #     V, coeff, idx = Vo, coeffo, idxo 
                # else:
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
                Vs.append(V)
                coeffs.append(coeff)
                idxs.append(idx)
                # print(V, coeff, idx)
                # op = [0]*self._n_qubits
                # for i in range(self._n_qubits):
                #     op[i] = Pauli.Z 
                # print(count)
                # return V
            else:
                V, coeff, idx = sampled[0][m], sampled[1][m], sampled[2][m]
                count = len(V)
            if cheat:
                # print(V)
                if self.noise:
                    evol, depths = U_drift(self.circuit.get_unitary(), rep=1, n=count, t_max=self.t_max, V=V, coeff=coeff, n_qubits=self._n_qubits, state=self._initial_state, measurement=measurement,error=self.noise_param)
                else:
                    evol = U_drift(self.circuit.get_unitary(), rep=1, n=count, t_max=self.t_max, V=V, coeff=coeff, n_qubits=self._n_qubits, state=self._initial_state, measurement=measurement)  
                    # print(evol)
                # evols[m].append(evol)
                # self.depth[m].append(depths)
            else:
                for n in range(0,min(count, self._rep+1)):

                    if n ==0:
                        circ = self.circuit.copy()
                        self.depth[m].append(0)
                    else:
                        if n == 1:
                            circ = self.circuit.copy() 
                        # for _ in range(n):
                            # print(V, coeff, idx)
                        for paulis in V[n]:
                            # p = self.Convert_String_to_Op(paulis)
                            if not self.noise:
                                # pbox = PauliExpBox(p, coeff[(n-1)*self.N+j])
                                # circ.add_pauliexpbox(pbox, np.arange(self._n_qubits))
                                # print(paulis)
                                circ.append(Noise_PauliGadget(paulis,coeff[n],0))  
                                # print(n)
                                # print(tk_to_qiskit(circ))
                            else:
                                circ.append(Noise_PauliGadget(paulis,coeff[n],self.noise_param))   
                        self.depth[m].append(self.depth[m][-1]+len(V[n])) 
                    # print(circ.depth(), circ.n_gates)
                    naive_circuit = circ.copy()
                    # Transform.DecomposeBoxes().apply(naive_circuit)
                    
                    # print(tk_to_qiskit(naive_circuit))
                    if spectral:
                        self.U_sims[m].append(naive_circuit.get_unitary())
                        
                    else:
                        # naive_circuit.measure_all()
                        # compiled_circuit = self.backend.get_compiled_circuit(naive_circuit)
                        # handle = self.backend.process_circuit(compiled_circuit, n_shots=100)
                        # counts = self.backend.get_result(handle).get_counts()
                        # self.shots.append(counts)
                        # statevec = self.statebackend.run_circuit(compiled_circuit).get_state()
                        statevec = naive_circuit.get_statevector()
                        if measurement=='H':
                            self.E.append((statevec.conj().T@self.H)@statevec)
                        elif measurement=='Z':
                            self.E.append((statevec.conj().T@self.Z)@statevec) 

        if spectral:
            if sampled == None:
                if cheat:
                    # return evols, [Vs, coeffs, idxs], self.depth
                    return evol
                else:
                    return self.U_sims, [Vs, coeffs, idxs], self.depth
            else:
                if cheat:
                    return evols, [Vs, coeffs, idxs], self.depth
                else:
                    return self.U_sims, [V, coeff, idx], self.depth
        else:
            # return count_parity(self.shots), [V, coeff, idx]
            if cheat:
                return evols, [Vs, coeffs, idxs], self.depth
            else:
                return self.E, [V, coeff, idx], self.depth


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