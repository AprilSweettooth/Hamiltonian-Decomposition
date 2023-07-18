from utils.plot import evol_plot, cheat_plot, compare_plot, exp_plot

from pytket.utils import QubitPauliOperator, gen_term_sequence_circuit
from pytket.circuit import Circuit, OpType
from pytket.extensions.qiskit import AerBackend, AerStateBackend
from pytket.transform import Transform
from pytket.partition import PauliPartitionStrat, GraphColourMethod

import numpy as np
from scipy.linalg import expm


class AlgorithmHamSimTrotter:
    """This class Trotterises the time evolution operator using scipy matrix multiplication
    It is very fast but does not use circuits and therefor should just be used for testing and developing
    Now Suzuki-trotter algorithm is also included to make comparison
    """

    def __init__(self,
                 initial_state: Circuit, 
                 qubit_operator: QubitPauliOperator,
                 measurements: "list[QubitPauliOperator]",
                 t_max: float,
                 n_trotter_steps: int,
                 symbol: str,
                 *args,
                 **kwargs):

        self.circuit = initial_state
        self._n_qubits = initial_state.n_qubits
        self._initial_state = AerStateBackend().run_circuit(initial_state).get_state()
        self._qubit_operator = qubit_operator
        self.rev = self._qubit_operator
        self._rev_qubit_op = QubitPauliOperator.from_list(self.rev.to_list()[::-1])
        self._n_trotter_step = n_trotter_steps
        self._time_step = t_max/n_trotter_steps
        self._time_space = np.linspace(0,t_max,n_trotter_steps+1)
        f = lambda m: m[0] if len(m) == 1 else m[0] + f(m[1:])
        self.m = measurements
        self._measurements_overall = f(measurements)
        self._measurements = [m.to_sparse_matrix(self._n_qubits) for m in measurements]
        self.sym = symbol
        self.backend = AerBackend()
        self.statebackend = AerStateBackend()
        self.gate_count = []
        self.exp = [] 
        self.infidelity = []
        op = self._qubit_operator
        op.subs({self.sym:1})
        self._trotter_step_m = expm(-1j * op.to_sparse_matrix(self._n_qubits).toarray() * self._time_step ) #exp(-i H tau)
        self._evolved_measurement = {}
        self._evolved_measurements = {}

    def _measure(self,trotter_evolution,split):
        #< Psi_t | O | Psi_t > loop over O
        if split:
            return [np.vdot(trotter_evolution, operator.dot(trotter_evolution)).real.item() for operator in self._measurements]
        #< Psi_0 | Psi_t >
        else:
            return abs(np.vdot(self._initial_state, trotter_evolution))**2

    def _trotter_step_cheat(self,split):
        for t in self._time_space:
            if t == 0:
                trotter_evolution = self._initial_state
            else:
                trotter_evolution = np.matmul(self._trotter_step_m, trotter_evolution)
            self._evolved_measurements[t] = self._measure(trotter_evolution, split)

    def execute(self, labels, cheat=True, split=False, plot=True):
        if cheat:
            self._trotter_step_cheat(split)
            if plot:
                cheat_plot(self._evolved_measurements, labels)
        else:
            if split:
                exp_plot(self._evolved_measurements, labels)
            else:
                evol_plot(self._time_space, self.exp, self.gate_count, self.infidelity, labels)
    
    def compare(self, exps, gates, infidelities, labels):
        compare_plot(self._time_space, exps, gates, infidelities, labels)

    def _trotter_step(self, n, reverse=False):
        ref_cir = Circuit(self._n_qubits)
        cir = self.circuit
        if reverse:
            ansatz_circuit = gen_term_sequence_circuit(
                self._qubit_operator, ref_cir, PauliPartitionStrat.CommutingSets, GraphColourMethod.Lazy
            )
            rev_ansatz_circuit = gen_term_sequence_circuit(
                self._rev_qubit_op, ref_cir, PauliPartitionStrat.CommutingSets, GraphColourMethod.Lazy
            )
            while n > 0:
                cir.append(ansatz_circuit)
                cir.append(rev_ansatz_circuit)
                n -= 1      
        else:
            ansatz_circuit = gen_term_sequence_circuit(
                self._qubit_operator, ref_cir, PauliPartitionStrat.CommutingSets, GraphColourMethod.Lazy
            )
            while n > 0:
                cir.append(ansatz_circuit)
                n -= 1   
        return cir
    
    def lie_trotter(self,return_value=False):  
        for n in range(0, self._n_trotter_step+1):
            c = self._trotter_step(n)
            naive_circuit = c.copy()
            Transform.DecomposeBoxes().apply(naive_circuit)
            self.gate_count.append(naive_circuit.n_gates_of_type(OpType.CX))
            if n==0:
                state_true = self._initial_state
                symbol_dict = {self.sym: 0}
            else:
                state_true = np.matmul(self._trotter_step_m, state_true)
                symbol_dict = {self.sym: self._time_step/n}
            naive_circuit.symbol_substitution(symbol_dict)
            compiled_circuit = self.statebackend.get_compiled_circuit(naive_circuit)
            statevec = self.statebackend.run_circuit(compiled_circuit).get_state()
            self.infidelity.append(1 - np.abs(np.conj(state_true).dot(statevec))**2)
            self.exp.append(abs(np.vdot(self._initial_state,statevec))**2)
            self._evolved_measurement[self._time_space[n]] = self._measurements_overall.state_expectation(statevec)
            self._evolved_measurements[self._time_space[n]] = [m.state_expectation(statevec) for m in self.m]  
        if return_value:
            return self.exp, self.gate_count, self.infidelity, self._evolved_measurements

    def second_order_suzuki_trotter(self,return_value=False):  
        for n in range(0, self._n_trotter_step+1):
            c = self._trotter_step(n, True)
            naive_circuit = c.copy()
            Transform.DecomposeBoxes().apply(naive_circuit)
            self.gate_count.append(naive_circuit.n_gates_of_type(OpType.CX))
            if n==0:
                state_true = self._initial_state
                symbol_dict = {self.sym: 0}
            else:
                state_true = np.matmul(self._trotter_step_m, state_true)
                symbol_dict = {self.sym: self._time_step/(2*n)}
            naive_circuit.symbol_substitution(symbol_dict)
            compiled_circuit = self.statebackend.get_compiled_circuit(naive_circuit)
            statevec = self.statebackend.run_circuit(compiled_circuit).get_state()
            self.infidelity.append(1 - np.abs(np.conj(state_true).dot(statevec))**2)
            self.exp.append(abs(np.vdot(self._initial_state,statevec))**2)
            self._evolved_measurement[self._time_space[n]] = self._measurements_overall.state_expectation(statevec)
            self._evolved_measurements[self._time_space[n]] = [m.state_expectation(statevec) for m in self.m]  
        if return_value:
            return self.exp, self.gate_count, self.infidelity, self._evolved_measurements

    def suzuki_trotter_cir_gen(self, n, order, high_coeff=1):
        if order % 2 == 1:
            print('Must be even order !')
            return False
        suzuki_circ = Circuit(self._n_qubits)
        if order == 2:
            cir = Circuit(self._n_qubits)
            ref_cir = Circuit(self._n_qubits)
            ansatz_circuit = gen_term_sequence_circuit(
                self._qubit_operator, ref_cir, PauliPartitionStrat.CommutingSets, GraphColourMethod.Lazy
            )
            rev_ansatz_circuit = gen_term_sequence_circuit(
                self._rev_qubit_op, ref_cir, PauliPartitionStrat.CommutingSets, GraphColourMethod.Lazy
            )
            cir.append(ansatz_circuit)
            cir.append(rev_ansatz_circuit)
            Transform.DecomposeBoxes().apply(cir)
            symbol_dict = {self.sym: high_coeff*self._time_step/(2*n)}
            cir.symbol_substitution(symbol_dict)
            return cir
            # return self.second_order_suzuki_trotter(high_coeff)
        '''
            References:
            D. Berry, G. Ahokas, R. Cleve and B. Sanders,
            "Efficient quantum algorithms for simulating sparse Hamiltonians" (2006).
            `arXiv:quant-ph/0508139 <https://arxiv.org/abs/quant-ph/0508139>`_
        '''
        s_k = (4 - 4**(1 / (order - 1)))**(-1)
        sub_2 = self.suzuki_trotter_cir_gen(n, order - 2, s_k)
        suzuki_circ.append(sub_2)
        suzuki_circ.append(sub_2)
        suzuki_circ.append(self.suzuki_trotter_cir_gen(n, order - 2, (1 - 4 * s_k)))
        suzuki_circ.append(sub_2)
        suzuki_circ.append(sub_2)
        return suzuki_circ
    
    def suzuki_trotter(self,order,return_value=False):
        gate_count = []
        exp = []    
        for n in range(0, self._n_trotter_step+1):
            if n==0:
                c = self.circuit
                state_true = self._initial_state
            else:
                c = self.circuit
                state_true = np.matmul(self._trotter_step_m, state_true)
                c_suzuki = self.suzuki_trotter_cir_gen(n,order)
                while n > 0:
                    c.append(c_suzuki)
                    n -= 1
            naive_circuit = c.copy()
            self.gate_count.append(naive_circuit.n_gates_of_type(OpType.CX))
            compiled_circuit = self.statebackend.get_compiled_circuit(naive_circuit)
            statevec = self.statebackend.run_circuit(compiled_circuit).get_state()
            self.infidelity.append(1 - np.abs(np.conj(state_true).dot(statevec))**2)
            self.exp.append(abs(np.vdot(self._initial_state,self.statebackend.run_circuit(compiled_circuit).get_state()))**2)
            self._evolved_measurement[self._time_space[n]] = self._measurements_overall.state_expectation(statevec)
            self._evolved_measurements[self._time_space[n]] = [m.state_expectation(statevec) for m in self.m]  
        if return_value:
            return self.exp, self.gate_count, self.infidelity, self._evolved_measurements