from utils.plot import evol_plot
from trotter.hamsimtrotter import AlgorithmHamSimTrotter

from pytket.utils import QubitPauliOperator
from pytket.pauli import Pauli, QubitPauliString
from pytket.circuit import Qubit, Circuit, OpType, PauliExpBox, fresh_symbol
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
                 qubit_operator: "list[QubitPauliOperator]",
                 coeff: list,
                #  measurements: "list[QubitPauliOperator]",
                 dict_qubit_operator: QubitPauliOperator,
                 t_max: float,
                 reps: int,
                 symbol: str,
                 *args,
                 **kwargs):

        self.circuit = initial_state
        self.c_copy = self.circuit.copy()
        self._n_qubits = initial_state.n_qubits
        self._initial_state = AerStateBackend().run_circuit(initial_state).get_state()
        self._qubit_operator = qubit_operator
        t=fresh_symbol("t")
        self.H = dict_qubit_operator
        self.H.subs({symbol:1})
        self.coeff = coeff
        self.t_max = t_max
        self._rep = reps
        self._time_step = t_max/self._rep
        self._time_space = np.linspace(0,t_max,self._rep+1)
        # self.m = measurements
        # f = lambda m: m[0] if len(m) == 1 else m[0] + f(m[1:])
        # self._measurements_overall = f(measurements)
        # self._measurements = [m.to_sparse_matrix(self._n_qubits) for m in measurements]
        self.backend = AerBackend()
        self.statebackend = AerStateBackend()
        self.gate_count = []
        self.exp = []  
        self.E = {}
        self.U_sim = np.zeros((2**self._n_qubits,2**self._n_qubits)) 

    def Drift_step(self, t):
        weights = [abs(co) for co in self.coeff]
        lambd = sum(weights)
        prob = [weight / lambd for weight in weights]
        op_index = list(range(0,len(weights)))
        sample_index = np.random.choice(op_index, self._rep, p=prob)
        sample_ops = [self._qubit_operator[s] for s in sample_index]
        new_coeff = lambd * (self.t_max/(np.pi/2)) / self._rep
        new_coeffs = [new_coeff*co/abs(co) for co in self.coeff]
        return sample_ops,new_coeffs,sample_index
    
    def qps_from_openfermion(self,paulis):
        """Convert OpenFermion tensor of Paulis to pytket QubitPauliString."""
        pauli_sym = {"I": Pauli.I, "X": Pauli.X, "Y": Pauli.Y, "Z": Pauli.Z}
        qlist = []
        plist = []
        for q, p in paulis:
            qlist.append(Qubit(q))
            plist.append(pauli_sym[p])
        return QubitPauliString(qlist, plist)

    def qpo_from_openfermion(self,openf_op):
        """Convert OpenFermion QubitOperator to pytket QubitPauliOperator."""
        tk_op = dict()
        for term, coeff in openf_op.terms.items():
            string = self.qps_from_openfermion(term)
            tk_op[string] = coeff
        return QubitPauliOperator(tk_op)
    
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

    def Drift_exp(self):
        for n in range(self._rep,self._rep+1):
            if n ==0:
                circ = self.circuit.copy()
            else:
                circ = self.circuit.copy() 
                V, coeff, idx = self.Drift_step(self._time_space[n])
                # print(V, coeff, idx)
                for j in range(n):
                    V_j = self.Convert_String_to_Op(V[j])
                    pbox = PauliExpBox(V_j, coeff[idx[j]])
                    circ.add_pauliexpbox(pbox, np.arange(self._n_qubits))

            naive_circuit = circ.copy()
            Transform.DecomposeBoxes().apply(naive_circuit)
            # print(tk_to_qiskit(naive_circuit))
            # self.gate_count.append(naive_circuit.n_gates_of_type(OpType.CX))
            self.U_sim = naive_circuit.get_unitary()
            # compiled_circuit = self.statebackend.get_compiled_circuit(naive_circuit)
            # statevec = self.statebackend.run_circuit(compiled_circuit).get_state()
            # self.exp.append(abs(np.vdot(self._initial_state,statevec))**2)
            # self.E[self._time_space[n]] = self.H.state_expectation(statevec, [Qubit(i) for i in range(self._n_qubits)]) 


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