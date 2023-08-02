from utils.plot import evol_plot, cheat_plot, compare_plot, exp_plot
from utils.func import factorial, calculate_error

from pytket.utils import QubitPauliOperator, gen_term_sequence_circuit
from pytket.circuit import Circuit, OpType, PauliExpBox, Qubit
from pytket.extensions.qiskit import AerBackend, AerStateBackend
from pytket.transform import Transform
from pytket.partition import PauliPartitionStrat, GraphColourMethod
from pytket.circuit.display import render_circuit_jupyter
from pytket.extensions.qiskit import tk_to_qiskit

from tqdm import tqdm
import numpy as np
import math as m
from scipy.linalg import expm
import itertools
from numpy.linalg import matrix_power


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
        self._initial_unitary = initial_state.copy().get_unitary()
        self._initial_state = AerStateBackend().run_circuit(initial_state.copy()).get_state()
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
        self.H = QubitPauliOperator.from_list(qubit_operator.to_list())
        self.H.subs({self.sym:1})
        op = QubitPauliOperator.from_list(qubit_operator.to_list())
        op.subs({self.sym:1})
        self.real_op = op
        # e = 1
        # for i in range(len(op.to_list())):
            # e *= expm(-1j * QubitPauliOperator.from_list([op.to_list()[i]]).to_sparse_matrix(self._n_qubits).toarray() )
        self._trotter_step_m = expm(-1j * op.to_sparse_matrix(self._n_qubits).toarray() * self._time_step ) #exp(-i H tau)
        # self._trotter_step_m = e ** self._time_step
        self._real_measurement = {}
        self._evolved_measurement = {}
        self._evolved_measurements = {}
        self.U_sim = np.zeros((2**self._n_qubits,2**self._n_qubits)) 
        self.U_sims = []       
        self.E = {}

    def _measure(self,trotter_evolution,exps):
        #< Psi_t | O | Psi_t > loop over O
        if exps=='split':
            return [np.vdot(trotter_evolution, operator.dot(trotter_evolution)).real.item() for operator in self._measurements]
        elif exps=='overall':
            return np.vdot(trotter_evolution, self._measurements_overall.to_sparse_matrix().dot(trotter_evolution)).real.item()
        #< Psi_0 | Psi_t >
        elif exps=='proj':
            return abs(np.vdot(self._initial_state, trotter_evolution))**2
        elif exps=='Energy':
            return np.vdot(trotter_evolution, self.H.to_sparse_matrix().dot(trotter_evolution))

    def _trotter_step_cheat(self,exps):
        e = expm(-1j * QubitPauliOperator.from_list([self.real_op.to_list()[0]]).to_sparse_matrix(self._n_qubits).toarray() * self._time_step) 
        for i in range(1,len(self.real_op.to_list())):
            e = np.matmul(e,expm(-1j * QubitPauliOperator.from_list([self.real_op.to_list()[i]]).to_sparse_matrix(self._n_qubits).toarray() ))
        for t in self._time_space:
            if t == 0:
                trotter_evolution = self._initial_state
            else:
                # print(self._trotter_step_m)
                trotter_evolution = np.matmul(self._trotter_step_m, trotter_evolution)
            self._real_measurement[t] = self._measure(trotter_evolution, exps)

    def execute(self, labels=['LieTrotter','LieTrotter','LieTrotter'], exps='proj', color='blue', cheat=True, plot=True):
        self._trotter_step_cheat(exps)
        if cheat:
            cheat_plot(self._real_measurement, labels, exps, color)
        else:
            if plot:
                if exps=='split':
                    exp_plot(self._real_measurement, self._evolved_measurements, labels, exps, color)
                elif exps=='overall':
                    exp_plot(self._real_measurement, self._evolved_measurement, labels, exps, color)
                elif exps=='Energy':
                    exp_plot(self._real_measurement, self.E, labels, exps, color) 
                elif exps=='proj':
                    evol_plot(self._real_measurement, self._time_space, self.exp, self.gate_count, self.infidelity, labels, color)
            else:
                # return self.U_sims, [matrix_power(self._trotter_step_m, s)@self._initial_unitary for s in range(self._n_trotter_step+1) ]
                return self.U_sim, matrix_power(self._trotter_step_m,(self._n_trotter_step+1))@self._initial_unitary

    def compare(self, exps, gates, infidelities, labels, colors):
        self._trotter_step_cheat(exps='proj')
        compare_plot(self._real_measurement, self._time_space, exps, gates, infidelities, labels, colors)

    def _trotter_step(self, n, reverse=False):
        ref_cir = Circuit(self._n_qubits)
        cir = self.circuit.copy()
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
            # pbox = PauliExpBox([Pauli.I, Pauli.X, Pauli.Y, Pauli.Z], 0.75)
            # a = Circuit(4).add_pauliexpbox(pbox, [0, 1, 2, 3])
      
            for i in range(n):
    
                cir.append(ansatz_circuit)
                # n -= 1   
          
        return cir
    
    def lie_trotter(self,return_value=False):  
        # change every 0 to trotter step for experimenting
        for n in range(0, self._n_trotter_step+1):
     
            c = self._trotter_step(n)
         
            naive_circuit = c.copy()
  
            Transform.DecomposeBoxes().apply(naive_circuit)

            # self.gate_count.append(naive_circuit.n_gates_of_type(OpType.CX))
            if n==0:
                # state_true = self._initial_state
                symbol_dict = {self.sym: 0}
            else:
                # state_true = np.matmul(self._trotter_step_m, state_true)
                symbol_dict = {self.sym: self._time_step/(np.pi/2)} 
            naive_circuit.symbol_substitution(symbol_dict)
            # print(tk_to_qiskit(naive_circuit))
            # print(tk_to_qiskit(naive_circuit))
            # print(naive_circuit.get_unitary())
            compiled_circuit = self.statebackend.get_compiled_circuit(naive_circuit)
            
            # print(compiled_circuit.get_commands())
            # self.U_sim = naive_circuit.get_unitary()
            # self.U_sims.append(self.U_sim)
            # print(self.U_sim)
            statevec = self.statebackend.run_circuit(compiled_circuit).get_state()
            # print(statevec)
            # self.infidelity.append(1 - np.abs(np.conj(state_true).dot(statevec))**2)
            # self.exp.append(abs(np.vdot(self._initial_state,statevec))**2)
            # self._evolved_measurement[self._time_space[n]] = self._measurements_overall.state_expectation(statevec)
            # self._evolved_measurements[self._time_space[n]] = [m.state_expectation(statevec) for m in self.m]  
            self.E[self._time_space[n]] = self.H.state_expectation(statevec, [Qubit(i) for i in range(self._n_qubits)])
        if return_value:
            return self.exp, self.gate_count, self.infidelity, self._evolved_measurements

    def second_order_suzuki_trotter(self,return_value=False):  
        for n in range(0, self._n_trotter_step+1):
            c = self._trotter_step(n, True)
            naive_circuit = c.copy()
            Transform.DecomposeBoxes().apply(naive_circuit)
            # self.gate_count.append(naive_circuit.n_gates_of_type(OpType.CX))
            if n==0:
                # state_true = self._initial_state
                symbol_dict = {self.sym: 0}
            else:
                # state_true = np.matmul(self._trotter_step_m, state_true)
                symbol_dict = {self.sym: self._time_step/(np.pi)}
            naive_circuit.symbol_substitution(symbol_dict)
            # print(tk_to_qiskit(naive_circuit))
            # print(naive_circuit.get_commands())
            compiled_circuit = self.statebackend.get_compiled_circuit(naive_circuit)
            # self.U_sim = naive_circuit.get_unitary()
            statevec = self.statebackend.run_circuit(compiled_circuit).get_state()
            # self.infidelity.append(1 - np.abs(np.conj(state_true).dot(statevec))**2)
            # self.exp.append(abs(np.vdot(self._initial_state,statevec))**2)
            # self._evolved_measurement[self._time_space[n]] = self._measurements_overall.state_expectation(statevec)
            # self._evolved_measurements[self._time_space[n]] = [m.state_expectation(statevec) for m in self.m] 
            self.E[self._time_space[n]] = self.H.state_expectation(statevec, [Qubit(i) for i in range(self._n_qubits)]) 
        if return_value:
            return self.exp, self.gate_count, self.infidelity, self._evolved_measurements

    def suzuki_trotter_cir_gen(self, order, high_coeff=1, perm=False, perm_op=None, rev_perm_op=None):
        if order % 2 == 1:
            print('Must be even order !')
            return False
        suzuki_circ = Circuit(self._n_qubits)
        if order == 2:
            cir = Circuit(self._n_qubits)
            ref_cir = Circuit(self._n_qubits)
            if not perm:
                ansatz_circuit = gen_term_sequence_circuit(
                    self._qubit_operator, ref_cir, PauliPartitionStrat.CommutingSets, GraphColourMethod.Lazy
                )
                rev_ansatz_circuit = gen_term_sequence_circuit(
                    self._rev_qubit_op, ref_cir, PauliPartitionStrat.CommutingSets, GraphColourMethod.Lazy
                )
            else:
                ansatz_circuit = gen_term_sequence_circuit(
                    perm_op, ref_cir, PauliPartitionStrat.CommutingSets, GraphColourMethod.Lazy
                )
                rev_ansatz_circuit = gen_term_sequence_circuit(
                    rev_perm_op, ref_cir, PauliPartitionStrat.CommutingSets, GraphColourMethod.Lazy
                ) 
            cir.append(ansatz_circuit)
            cir.append(rev_ansatz_circuit)
            Transform.DecomposeBoxes().apply(cir)
            symbol_dict = {self.sym: high_coeff*self._time_step/(np.pi)}
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
        sub_2 = self.suzuki_trotter_cir_gen(order - 2, s_k*high_coeff)
        suzuki_circ.append(sub_2)
        suzuki_circ.append(sub_2)
        suzuki_circ.append(self.suzuki_trotter_cir_gen(order - 2, (1 - 4 * s_k)*high_coeff))
        suzuki_circ.append(sub_2)
        suzuki_circ.append(sub_2)
        return suzuki_circ
    
    def suzuki_trotter(self,order,return_value=False):
        for n in range(self._n_trotter_step, self._n_trotter_step+1):
            if n==0:
                c = self.circuit.copy()
                # state_true = self._initial_state
            else:
                c = self.circuit.copy()
                # state_true = np.matmul(self._trotter_step_m, state_true)
                c_suzuki = self.suzuki_trotter_cir_gen(order)
                while n > 0:
                    
                    c.append(c_suzuki)
                    n -= 1
           
            naive_circuit = c.copy()
            # print(tk_to_qiskit(naive_circuit)) 
            # self.gate_count.append(naive_circuit.n_gates_of_type(OpType.CX))
            # compiled_circuit = self.statebackend.get_compiled_circuit(naive_circuit)
            self.U_sim = naive_circuit.get_unitary()
        #     statevec = self.statebackend.run_circuit(compiled_circuit).get_state()
        #     self.infidelity.append(1 - np.abs(np.conj(state_true).dot(statevec))**2)
        #     self.exp.append(abs(np.vdot(self._initial_state,self.statebackend.run_circuit(compiled_circuit).get_state()))**2)
        #     self._evolved_measurement[self._time_space[n]] = self._measurements_overall.state_expectation(statevec)
        #     self._evolved_measurements[self._time_space[n]] = [m.state_expectation(statevec) for m in self.m]  
        # if return_value:
        #     return self.exp, self.gate_count, self.infidelity, self._evolved_measurements
        

    def prod(self, q, k):
        p = 1
        for j in range(1,k+1):
            if j != q:
                p *= q**2/(q**2-j**2)
            else:
                p *= 1
        return p

    def multi_prod_trotter_cir_gen(self, order):
        if order % 2 == 1:
            print('Must be even order !')
            return False
        k = len(self._qubit_operator.to_list())
        mulprod_circs = [Circuit(self._n_qubits)]*(k)
        gamma_exp = m.ceil(np.exp((k)*(1+np.log(0.3081)/2+np.log(((2*(k-1))**2.5)/8)/(2*(k-1)))))
        Cs = np.zeros((2**self._n_qubits,), dtype='complex128')
        for q in range(0,k):
            # Cs[q] = (((q+1)**2/((q+1)**2-gamma_exp**2)) * self.prod((q+1),k-1))
            Cs[q] = self.prod((q+1),k)
            i = q+1
            while i>0:
                mulprod_circs[q].append(self.suzuki_trotter_cir_gen(order, 1/(q+1)))
                i -= 1
        # Cs[-1] = self.prod(gamma_exp**2,k)
        # for i in range(k):
        #     mulprod_circs[-1].append(self.suzuki_trotter_cir_gen(order, 1/gamma_exp))
        return mulprod_circs, Cs
    
    def LCU(self,order):  
        for n in range(self._n_trotter_step, self._n_trotter_step+1):
            if n==0:
                cs = [self.circuit.copy()] * (len(self._qubit_operator.to_list()))
                coeff = np.ones((2**self._n_qubits,), dtype='complex128')
                # state_true = self._initial_state
            else:
                cs = [self.circuit.copy()] * (len(self._qubit_operator.to_list()))
                c_lcu, coeff = self.multi_prod_trotter_cir_gen(order)
                # state_true = np.matmul(self._trotter_step_m, state_true)
                while n > 0:
                    for j in range(len(cs)):
                        cs[j].append(c_lcu[j])
                    n -= 1
            # statevec = np.zeros((2**self._n_qubits,), dtype='complex128')
            # self.gate_count.append((cs[0].n_gates_of_type(OpType.CX)))
            for i in range((len(self._qubit_operator.to_list()))): 
                naive_circuit = cs[i].copy()
                # compiled_circuit = self.statebackend.get_compiled_circuit(naive_circuit)
                self.U_sim += coeff[i]*naive_circuit.get_unitary()
            #     statevec += coeff[i]*self.statebackend.run_circuit(compiled_circuit).get_state()
            # self.infidelity.append(1 - np.abs(np.conj(state_true).dot(statevec))**2)
            # self.exp.append(abs(np.vdot(self._initial_state,statevec))**2)
            # self._evolved_measurement[self._time_space[n]] = self._measurements_overall.state_expectation(statevec)
            # self._evolved_measurements[self._time_space[n]] = [m.state_expectation(statevec) for m in self.m]   

    def random_perm(self,order):
        perms = list(itertools.permutations(self._qubit_operator.to_list()))
        L = len(self._qubit_operator.to_list())
        for n in range(self._n_trotter_step, self._n_trotter_step+1):
            # statevec = np.zeros((2**self._n_qubits,), dtype='complex128')
            # if n==0:
                # statevec = state_true = self._initial_state
                # self.exp.append(1)
                # self.gate_count.append(1)
                # self.infidelity.append(1 - np.abs(np.conj(state_true).dot(statevec))**2)
                # self._evolved_measurement[self._time_space[n]] = self._measurements_overall.state_expectation(statevec)
                # self._evolved_measurements[self._time_space[n]] = [m.state_expectation(statevec) for m in self.m] 
            # else:
                # state_true = np.matmul(self._trotter_step_m, state_true)

            # for l in range(L):
            # m = n
            op = QubitPauliOperator.from_list(perms[np.random.randint(L, size=1)])
            rev_op = QubitPauliOperator.from_list(op.to_list()[::-1])
            c = self.circuit.copy()
            c_perm = self.suzuki_trotter_cir_gen(order, perm=True, perm_op=op, rev_perm_op=rev_op)
            while n > 0:
                c.append(c_perm)
                n -= 1
            naive_circuit = c.copy()
            # compiled_circuit = self.statebackend.get_compiled_circuit(naive_circuit)
            self.U_sim = naive_circuit.get_unitary()

                    # statevec += self.statebackend.run_circuit(compiled_circuit).get_state()
            # self.U_sim = self.U_sim/(factorial(L))
                # self.gate_count.append(1)
                # self.exp.append(abs(np.vdot(self._initial_state,statevec/(factorial(L))))**2)
                # self.infidelity.append(1 - np.abs(np.conj(state_true).dot(statevec))**2)
                # self._evolved_measurement[self._time_space[n]] = self._measurements_overall.state_expectation(statevec)
                # self._evolved_measurements[self._time_space[n]] = [m.state_expectation(statevec) for m in self.m]  
        