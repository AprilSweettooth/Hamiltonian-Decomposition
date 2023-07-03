# # Expectation Values

# Given a circuit generating a quantum state $\lvert \psi \rangle$, it is very common to have an operator $H$ and ask for the expectation value $\langle \psi \vert H \vert \psi \rangle$. A notable example is in quantum computational chemistry, where $\lvert \psi \rangle$ encodes the wavefunction for the electronic state of a small molecule, and the energy of the molecule can be derived from the expectation value with respect to the molecule's Hamiltonian operator $H$.
#
# This example uses this chemistry scenario to demonstrate the overall procedure for using `pytket` to perform advanced high-level procedures. We build on top of topics covered by several other example notebooks, including circuit generation, optimisation, and using different backends.
#
# There is limited built-in functionality in `pytket` for obtaining expectation values from circuits. This is designed to encourage users to consider their needs for parallelising the processing of circuits, manipulating results (e.g. filtering, adjusting counts to mitigate errors, and other forms of data processing), or more advanced schemes for grouping the terms of the operator into measurement circuits. For this example, suppose that we want to focus on reducing the queueing time for IBM device backends, and filter our shots to eliminate some detected errors.
#
# This notebook makes use of the Qiskit and ProjectQ backend modules `pytket_qiskit` and `pytket_projectq`, as well as the electronic structure module `openfermion`, all three of which should first be installed via `pip`.
#
# We will start by generating an ansatz and Hamiltonian for the chemical of interest. Here, we are just using a simple model of $\mathrm{H}_2$ with four qubits representing the occupation of four spin orbitals.

from pytket import Circuit, Qubit, Bit
from sympy import symbols

# Generate ansatz and Hamiltonian:

ansatz = Circuit()
qubits = ansatz.add_q_register("q", 4)
args = symbols("a0 a1 a2 a3 a4 a5 a6 a7")
for i in range(4):
    ansatz.Ry(args[i], qubits[i])
for i in range(3):
    ansatz.CX(qubits[i], qubits[i + 1])
for i in range(4):
    ansatz.Ry(args[4 + i], qubits[i])
ansatz.measure_all()

for command in ansatz:
    print(command)

# In reality, you would use an expectation value calculation as the objective function for a classical optimisation routine to determine the parameter values for the ground state. For the purposes of this notebook, we will use some predetermined values for the ansatz, already optimised for $\mathrm{H}_2$.

arg_values = [
    7.17996183e-02,
    2.95442468e-08,
    1.00000015e00,
    1.00000086e00,
    9.99999826e-01,
    1.00000002e00,
    9.99999954e-01,
    1.13489747e-06,
]

ansatz.symbol_substitution(dict(zip(args, arg_values)))

# We can use for example the openfermion library to express an Hamiltonian as a sum of tensors of paulis.
import openfermion as of

hamiltonian = (
    -0.0970662681676282 * of.QubitOperator("")
    + -0.045302615503799284 * of.QubitOperator("X0 X1 Y2 Y3")
    + 0.045302615503799284 * of.QubitOperator("X0 Y1 Y2 X3")
    + 0.045302615503799284 * of.QubitOperator("Y0 X1 X2 Y3")
    + -0.045302615503799284 * of.QubitOperator("Y0 Y1 X2 X3")
    + 0.17141282644776884 * of.QubitOperator("Z0")
    + 0.16868898170361213 * of.QubitOperator("Z0 Z1")
    + 0.12062523483390425 * of.QubitOperator("Z0 Z2")
    + 0.16592785033770352 * of.QubitOperator("Z0 Z3")
    + 0.17141282644776884 * of.QubitOperator("Z1")
    + 0.16592785033770352 * of.QubitOperator("Z1 Z2")
    + 0.12062523483390425 * of.QubitOperator("Z1 Z3")
    + -0.22343153690813597 * of.QubitOperator("Z2")
    + 0.17441287612261608 * of.QubitOperator("Z2 Z3")
    + -0.22343153690813597 * of.QubitOperator("Z3")
)

# This can be converted into pytket's QubitPauliOperator type.
#
# The OpenFermion `QubitOperator` class represents the operator by its decomposition into a linear combination of Pauli operators (tensor products of the $I$, $X$, $Y$, and $Z$ matrices).
#
# A `QubitPauliString` is a sparse representation of a Pauli operator with support over some subset of qubits.

from pytket.pauli import Pauli, QubitPauliString
from pytket.utils.operators import QubitPauliOperator

pauli_sym = {"I": Pauli.I, "X": Pauli.X, "Y": Pauli.Y, "Z": Pauli.Z}


def qps_from_openfermion(paulis):
    """Convert OpenFermion tensor of Paulis to pytket QubitPauliString."""
    qlist = []
    plist = []
    for q, p in paulis:
        qlist.append(Qubit(q))
        plist.append(pauli_sym[p])
    return QubitPauliString(qlist, plist)


def qpo_from_openfermion(openf_op):
    """Convert OpenFermion QubitOperator to pytket QubitPauliOperator."""
    tk_op = dict()
    for term, coeff in openf_op.terms.items():
        string = qps_from_openfermion(term)
        tk_op[string] = coeff
    return QubitPauliOperator(tk_op)


hamiltonian_op = qpo_from_openfermion(hamiltonian)

# We can simulate this exactly using a statevector simulator like ProjectQ. This has a built-in method for fast calculations of expectation values that works well for small examples like this.

from pytket.extensions.projectq import ProjectQBackend

backend = ProjectQBackend()
ideal_energy = backend.get_operator_expectation_value(ansatz, hamiltonian_op)
print(ideal_energy)

# Ideally the state generated by this ansatz will only span the computational basis states with exactly two of the four qubits in state $\lvert 1 \rangle$. This is because these basis states correspond to two electrons being present in the molecule.
#
# This ansatz is a hardware-efficient model that is designed to explore a large portion of the Hilbert space with relatively few entangling gates. Unfortunately, with this much freedom, it will regularly generate states that have no physical interpretation such as states spanning multiple basis states corresponding to different numbers of electrons in the system (which we assume is fixed and conserved).
#
# We can mitigate this by using a syndrome qubit that calculates the parity of the other qubits. Post-selecting this syndrome with $\langle 0 \rvert$ will project the remaining state onto the subspace of basis states with even parity, increasing the likelihood the observed state will be a physically admissible state.
#
# Even if the ansatz parameters are tuned to give a physical state, real devices have noise and imperfect gates, so in practice we may also measure bad states with a small probability. If this syndrome qubit is measured as 1, it means an error has definitely occurred, so we should discard the shot.

syn = Qubit("synq", 0)
syn_res = Bit("synres", 0)
ansatz.add_qubit(syn)
ansatz.add_bit(syn_res)
for qb in qubits:
    ansatz.CX(qb, syn)
ansatz.Measure(syn, syn_res)

# Using this, we can define a filter function which removes the shots which the syndrome qubit detected as erroneous. `BackendResult` objects allow retrieval of shots in any bit order, so we can retrieve the `synres` results separately and use them to filter the shots from the remaining bits. The Backends example notebook describes this in more detail.

from collections import Counter


def filter_shots(backend_result, syn_res_bit):
    bits = sorted(backend_result.get_bitlist())
    bits.remove(syn_res_bit)
    syn_shots = backend_result.get_shots([syn_res])[:, 0]
    main_shots = backend_result.get_shots(bits)
    return main_shots[syn_shots == 0]


def filter_counts(backend_result, syn_res_bit):
    bits = sorted(backend_result.get_bitlist())
    syn_index = bits.index(syn_res_bit)
    counts = backend_result.get_counts()
    filtered_counts = Counter()
    for readout, count in counts.items():
        if readout[syn_index] == 0:
            filtered_readout = tuple(v for i, v in enumerate(readout) if i != syn_index)
            filtered_counts[filtered_readout] += count
    return filtered_counts


# Depending on which backend we will be using, we will need to compile each circuit we run to conform to the gate set and connectivity constraints. We can define a compilation pass for each backend that optimises the circuit and maps it onto the backend's gate set and connectivity constraints. We don't expect this to change our circuit too much as it is already near-optimal.

from pytket.passes import OptimisePhaseGadgets, SequencePass


def compiler_pass(backend):
    return SequencePass([OptimisePhaseGadgets(), backend.default_compilation_pass()])


# Given the full statevector, the expectation value can be calculated simply by matrix multiplication. However, with a real quantum system, we cannot observe the full statevector directly. Fortunately, the Pauli decomposition of the operator gives us a sequence of measurements we should apply to obtain the relevant information to reconstruct the expectation value.
#
# The utility method `append_pauli_measurement` takes a single term of a `QubitPauliOperator` (a `QubitPauliString`) and appends measurements in the corresponding bases to obtain the expectation value for that particular Pauli operator. We will want to make a new `Circuit` object for each of the measurements we wish to observe.
#

from pytket.predicates import CompilationUnit
from pytket.utils import append_pauli_measurement


def gen_pauli_measurement_circuits(state_circuit, compiler_pass, operator):
    # compile main circuit once
    state_cu = CompilationUnit(state_circuit)
    compiler_pass.apply(state_cu)
    compiled_state = state_cu.circuit
    final_map = state_cu.final_map
    # make a measurement circuit for each pauli
    pauli_circuits = []
    coeffs = []
    energy = 0
    for p, c in operator.terms.items():
        if p == ():
            # constant term
            energy += c
        else:
            # make measurement circuits and compile them
            pauli_circ = Circuit(state_circuit.n_qubits - 1)  # ignore syndrome qubit
            append_pauli_measurement(qps_from_openfermion(p), pauli_circ)
            pauli_cu = CompilationUnit(pauli_circ)
            compiler_pass.apply(pauli_cu)
            pauli_circ = pauli_cu.circuit
            init_map = pauli_cu.initial_map
            # map measurements onto the placed qubits from the state
            rename_map = {
                i: final_map[o] for o, i in init_map.items() if o in final_map
            }
            pauli_circ.rename_units(rename_map)
            state_and_measure = compiled_state.copy()
            state_and_measure.append(pauli_circ)
            pauli_circuits.append(state_and_measure)
            coeffs.append(c)
    return pauli_circuits, coeffs, energy


# We can now start composing these together to get our generalisable expectation value function. Passing all of our circuits to `process_circuits` allows them to be submitted to IBM Quantum devices at the same time, giving substantial savings in overall queueing time. Since the backend will cache any results from `Backend.process_circuits`, we will remove the results when we are done with them to prevent memory bloating when this method is called many times.

from pytket.utils import expectation_from_shots, expectation_from_counts


def expectation_value(state_circuit, operator, backend, n_shots):
    if backend.supports_expectation:
        circuit = state_circuit.copy()
        compiled_circuit = backend.get_compiled_circuit(circuit)
        return backend.get_operator_expectation_value(
            compiled_circuit, qpo_from_openfermion(operator)
        )
    elif backend.supports_shots:
        syn_res_index = state_circuit.bit_readout[syn_res]
        pauli_circuits, coeffs, energy = gen_pauli_measurement_circuits(
            state_circuit, compiler_pass(backend), operator
        )
        handles = backend.process_circuits(pauli_circuits, n_shots=n_shots)
        for handle, coeff in zip(handles, coeffs):
            res = backend.get_result(handle)
            filtered = filter_shots(res, syn_res)
            energy += coeff * expectation_from_shots(filtered)
            backend.pop_result(handle)
        return energy
    elif backend.supports_counts:
        syn_res_index = state_circuit.bit_readout[syn_res]
        pauli_circuits, coeffs, energy = gen_pauli_measurement_circuits(
            state_circuit, compiler_pass(backend), operator
        )
        handles = backend.process_circuits(pauli_circuits, n_shots=n_shots)
        for handle, coeff in zip(handles, coeffs):
            res = backend.get_result(handle)
            filtered = filter_counts(res, syn_res)
            energy += coeff * expectation_from_counts(filtered)
            backend.pop_result(handle)
        return energy
    else:
        raise NotImplementedError("Implementation for state to be written")


# ...and then run it for our ansatz. `AerBackend` supports faster expectation value from snapshopts (using the `AerBackend.get_operator_expectation_value` method), but this only works when all the qubits in the circuit are default register qubits that go up from 0. So we will need to rename `synq`.

from pytket.extensions.qiskit import IBMQEmulatorBackend, AerBackend

ansatz.rename_units({Qubit("synq", 0): Qubit("q", 4)})

print(expectation_value(ansatz, hamiltonian, AerBackend(), 8000))
# Try replacing IBMQEmulatorBackend with IBMQBackend to submit the circuits to a real IBM Quantum device.
print(expectation_value(ansatz, hamiltonian, IBMQEmulatorBackend("ibmq_manila"), 8000))

# For basic practice with using pytket backends and their results, try editing the code here to:
# * Extend `expectation_value` to work with statevector backends (e.g. `AerStateBackend`)
# * Remove the row filtering from `filter_shots` and see the effect on the expectation value on a noisy simulation/device
# * Adapt `filter_shots` to be able to filter a counts dictionary and adapt `expectation_value` to calulate the result using the counts summary from the backend (`pytket.utils.expectation_from_counts` will be useful here)
