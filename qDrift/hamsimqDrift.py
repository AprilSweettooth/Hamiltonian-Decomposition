from utils.plot import evol_plot
from utils.func import count_parity
from trotter.hamsimtrotter import AlgorithmHamSimTrotter

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
                 *args,
                 **kwargs):

        self.circuit = initial_state
    
        self._n_qubits = initial_state.n_qubits
        self._initial_state = AerStateBackend().run_circuit(initial_state).get_state()
        self._qubit_operator = qubit_operator
   
        self.coeff = coeff
        self.t_max = t_max
        self._rep = reps
        self._time_step = t_max/self._rep
        self._time_space = np.linspace(0,t_max,self._rep+1)

        self.backend = AerBackend()
        self.statebackend = AerStateBackend()
        self.U_sims = []
        self.shots = []
        self.terms = 0
        self.p = None
        # self.U_sim = np.zeros((2**self._n_qubits,2**self._n_qubits)) 

    def Drift_step(self):
        weights = []
        for i in range(len(self.coeff)):
            if isinstance(self.coeff[i],list):
                weights.append(np.mean([self.coeff[i]]))
            else:
                raise Exception("Sorry, please give as lists of list")
        abs_weights = [abs(w) for w in weights]
        lambd = sum(abs_weights)
        prob = [weight / lambd for weight in abs_weights]
        self.p = prob
        op_index = list(range(0,len(weights)))
        sample_index = np.random.choice(op_index, self._rep, p=prob)
        sample_ops = [self._qubit_operator[s] for s in sample_index]
        new_coeff = lambd * (self.t_max/(np.pi/2)) / self._rep
        new_coeffs = [new_coeff*co/abs(co) for co in weights]
        return sample_ops,new_coeffs,sample_index
    
    def Convert_String_to_Op(self,paulis):
        ops = [0]*self._n_qubits
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

    def Drift_exp(self, track_no_paulistr=False, sampled=None):
        if sampled != None:
            V, coeff, idx = sampled[0], sampled[1], sampled[2]
        else:
            V, coeff, idx = self.Drift_step()
        # print(V)
        if track_no_paulistr:
            for v in V:
                self.terms += len(v)
        for n in range(0,self._rep+1):
            if n ==0:
                circ = self.circuit.copy()
            else:
                circ = self.circuit.copy() 
                
                # print(V, coeff, idx)
                for j in range(n):
                    for paulis in V[j]:
                        p = self.Convert_String_to_Op(paulis)
                        pbox = PauliExpBox(p, coeff[idx[j]])
                        circ.add_pauliexpbox(pbox, np.arange(self._n_qubits))

            naive_circuit = circ.copy()
            Transform.DecomposeBoxes().apply(naive_circuit)
            
            # print(tk_to_qiskit(naive_circuit))

            # self.U_sims.append(naive_circuit.get_unitary())

            naive_circuit.measure_all()
            compiled_circuit = self.backend.get_compiled_circuit(naive_circuit)
            handle = self.backend.process_circuit(compiled_circuit, n_shots=100)
            counts = self.backend.get_result(handle).get_counts()
            self.shots.append(counts)

            # statevec = self.statebackend.run_circuit(compiled_circuit).get_state()
            # self.exp.append(abs(np.vdot(self._initial_state,statevec))**2)
            # self.E[self._time_space[n]] = self.H.state_expectation(statevec, [Qubit(i) for i in range(self._n_qubits)]) 
        # return self.U_sims, V
        return count_parity(self.shots), idx, self.p


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