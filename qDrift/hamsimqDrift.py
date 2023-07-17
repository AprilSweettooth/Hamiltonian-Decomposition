from utils.plot import evol_plot, cheat_plot

from pytket.utils import QubitPauliOperator, gen_term_sequence_circuit
from pytket.pauli import Pauli, QubitPauliString
from pytket.circuit import Qubit, Circuit, OpType, fresh_symbol
from pytket.extensions.qiskit import AerBackend, AerStateBackend
from pytket.transform import Transform
from pytket.partition import PauliPartitionStrat, GraphColourMethod

import numpy as np
import math as m
from random import choice

class QDrift:
    """ The QDrift Trotterization method, which selects each each term in the
    Trotterization randomly, with a probability proportional to its weight. Based on the work
    of Earl Campbell in https://arxiv.org/abs/1811.08017.
    """
    def __init__(self,
                 initial_state: Circuit, 
                 qubit_operator: list,
                 coeff: list,
                 measurements: "list[QubitPauliOperator]",
                 t_max: float,
                 precision: float,
                 *args,
                 **kwargs):

        self.circuit = initial_state
        self._n_qubits = initial_state.n_qubits
        self._initial_state = AerStateBackend().run_circuit(initial_state).get_state()
        self._qubit_operator = qubit_operator
        self.coeff = coeff
        self.t_max = t_max
        self.precision = precision
        f = lambda m: m[0] if len(m) == 1 else m[0] + f(m[1:])
        self._measurements_overall = f(measurements)
        self._measurements = [m.to_sparse_matrix(self._n_qubits) for m in measurements]
        self.backend = AerBackend()
        self.statebackend = AerStateBackend()
        self.gate_count = []
        self.exp = []  

    def Drift_V(self, reps):
        weights = [abs(co) for co in self.coeff]
        lambd = sum(weights)
        prob = weights / lambd
        N = m.ceil(2 * (lambd ** 2) * (self.t_max ** 2) / self.precision)
        sample_ops = choice(self._qubit_operator, N, prob)
        new_coeff = lambd * self.t_max / N

        return sample_ops, new_coeff