from numpy.linalg import matrix_power
import scipy.sparse.linalg as ssl
import numpy as np
from scipy.linalg import expm
from openfermion import QubitOperator as q
from openfermion import get_sparse_operator
from utils.term_grouping import get_openfermion_str
import functools as ft

def get_Hmatrix(V,n,co):
    Hq = q()

    cirq_ops = []
    for op in V:
        cirq_ops.append(get_openfermion_str(op,n))

    for co, op in zip(co, cirq_ops):
        Hq += q(op, co)

    H_matrix = get_sparse_operator(Hq)
    return H_matrix

def U_exc(init, n, t_max, H_matrix, final=False):
    t_step = t_max/n
    if final:
        return ssl.expm(-1j * H_matrix * t_max).toarray()@init
    # print(U_mat.shape)
    else:
        U_mat = ssl.expm(-1j * H_matrix * t_step).toarray()
        evol = [ matrix_power(U_mat,i)@init for i in range(n+1) ]
        return evol


# Define the Pauli operator matrices
pauli_matrices = {
    'X': np.array([[0, 1], [1, 0]]),
    'Y': np.array([[0, -1j], [1j, 0]]),
    'Z': np.array([[1, 0], [0, -1]]),
    'I': np.array([[1, 0], [0, 1]])
}

def pauli_string_to_matrix(pauli_string, coeff=1):
    # Split the Pauli string into coefficient and operator parts
    coefficient, operators = coeff, pauli_string
    
    # Convert coefficient to a float
    coefficient = float(coefficient)
    
    # Initialize the resulting matrix as the coefficient times the identity matrix
    result_matrix = coefficient * pauli_matrices[pauli_string[0]] 
    
    # Multiply the matrix with each Pauli operator raised to the corresponding power
    for op in operators[1:]:
        result_matrix = np.kron(result_matrix, pauli_matrices[op])
    
    return result_matrix

# # Example usage
# pauli_str = '0.5XYZ'
# matrix_representation = pauli_string_to_matrix(pauli_str)
# print(matrix_representation)

def tensor_product_rz(angles):
    num_gates = len(angles)
    rz_matrices = [np.array([[np.exp(1j * angle / 2), 0], [0, np.exp(-1j * angle / 2)]]) for angle in angles]

    tensor_product = rz_matrices[0]
    for i in range(1, num_gates):
        tensor_product = np.kron(tensor_product, rz_matrices[i])

    return tensor_product

def unitary_to_statevector(unitary_matrix):
    num_qubits = int(np.log2(unitary_matrix.shape[0]))
    initial_state = np.zeros((2**num_qubits,1))
    initial_state[0, 0] = 1.0  # Set the initial state to |0...0>

    # statevector = np.dot(init, expm(-1j * np.angle(np.linalg.det(unitary_matrix)) / 2) * unitary_matrix)

    return unitary_matrix@initial_state

def replacer(s, index, newstring):
    return s[:index] + newstring + s[index + 1:]

# Define Pauli gate matrices
pauli_X = np.array([[0, 1], [1, 0]])
pauli_Y = np.array([[0, -1j], [1j, 0]])
pauli_Z = np.array([[1, 0], [0, -1]])

pauli_gates = [pauli_X, pauli_Y, pauli_Z]


def apply_depolarizing_error(matrix, error_rate):
    num_qubits = int(np.log2(matrix.shape[0]))
    error_matrix = np.sqrt(1 - 3/4 * error_rate) * np.eye(2 ** num_qubits, dtype='complex128')
    # for i in range(1, 4):
    #     pauli_i = np.eye(2)
    #     pauli_vector = [np.kron(pauli_i, pauli_i)]

    #     for j in range(3):
    #         if (i >> j) & 1:
    #             pauli_vector.append(np.kron(pauli_i, pauli_gates[j]))
    #     print(pauli_vector)
    error_matrix -= np.sqrt(error_rate / 4) * np.sum(pauli_gates, axis=0)

    noisy_matrix = np.matmul(error_matrix, matrix)
    return noisy_matrix

def add_depolarizing_error(circuit_matrix, num_qubits, error_rate):
    num_states = 2 ** num_qubits
    identity_matrix = np.eye(num_states)
    error_matrix = np.sqrt(1 - 3*error_rate/4) * identity_matrix + np.sqrt(error_rate / 4) * np.random.randn(num_states, num_states)
    noisy_circuit_matrix = np.dot(error_matrix, circuit_matrix)
    return noisy_circuit_matrix

def U_drift(init, rep, n, t_max, V, coeff, n_qubits,state, measurement=None, error=None):
    # t_step = t_max/rep
    # evol = []
    # m = init
    # for j in range(1,n+1):
    #     for i in range(1):
    #         # print((j-1)*rep+i)
    #         for v in V[(j-1)*1+i]:
    #             if error != None:
    #                 result_matrix =  apply_depolarizing_error(pauli_matrices[v[0]],error)
    #                 for op in v[1:]:
    #                     result_matrix = np.kron(result_matrix, apply_depolarizing_error(pauli_matrices[op],error))
    #                 identity = np.eye(init.shape[0],init.shape[1])
    #                 m = (np.cos(coeff[(j-1)*1+i] * np.pi/4 * t_step)*identity - 1j*np.sin(coeff[(j-1)*1+i] * np.pi/4 * t_step)*result_matrix )@m 
    #             else:
    #                 result_matrix =  pauli_matrices[v[0]] 
    #                 for op in v[1:]:
    #                     result_matrix = np.kron(result_matrix, pauli_matrices[op])
    #                 identity = np.eye(init.shape[0],init.shape[1])
    #                 m = (np.cos(coeff[(j-1)*1+i] * np.pi/4 * t_step)*identity - 1j*np.sin(coeff[(j-1)*1+i] * np.pi/4 * t_step)*result_matrix )@m
    #         evol.append(m)
    #     # depth.append(depth[-1]+len(V[(j-1)*rep+i]))
    #     # print(m)
    # return evol
    # print(1)
    evol = []
    depth = [0]
    evol.append(init)
    t_step = t_max/rep
    m = init
    # print(coeff)
    for j in range(1,n+1):
        for i in range(1):
            # print((j-1)*rep+i)
            for v in V[(j-1)*1+i]:
                # print(2)
                if error != None:
                    # print(v)
                    result_matrix =  apply_depolarizing_error(pauli_matrices[v[0]],error)
                    for op in v[1:]:
                        result_matrix = np.kron(result_matrix, apply_depolarizing_error(pauli_matrices[op],error))
                    identity = np.eye(init.shape[0],init.shape[1])
                    m = (np.cos(coeff[(j-1)*1+i] * np.pi/4 * t_step)*identity - 1j*np.sin(coeff[(j-1)*1+i] * np.pi/4 * t_step)*result_matrix )@m 
                else:
                    result_matrix =  pauli_matrices[v[0]] 
                    for op in v[1:]:
                        # print('here')
                        result_matrix = np.kron(result_matrix, pauli_matrices[op])
                    # if v[1]=='X':
                    #     for j in range(len(result_matrix)):
                    #         print(result_matrix[j])
                    #     print(v)
                    identity = np.eye(init.shape[0],init.shape[1])
                    m = (np.cos(coeff[(j-1)*1+i] * np.pi/4 * t_step)*identity - 1j*np.sin(coeff[(j-1)*1+i] * np.pi/4 * t_step)*result_matrix )@m
                    # print(op)
                    # if 'X' in op:
                    #     print(result_matrix)
                    # print(m)
        # print(V[(j-1)*1+i])
        depth.append(depth[-1]+len(V[(j-1)*1+i]))
        # print(m)
        evol.append(m)
    if measurement==None:
        return evol, depth
        # return evol[-1]
    elif isinstance(measurement,str):
        statevecs = [unitary_to_statevector(unitary) for unitary in evol]
        Z = sum([ft.reduce(np.kron,[np.array([[1, 0], [0, 1]])]*n_qubits) for _ in range(6)]) - sum([pauli_string_to_matrix(replacer('I'*n_qubits, idx, 'Z')) for idx in range(6)])
        return [(statevec.conj().T@Z)@statevec for statevec in statevecs], depth
    else:
        statevecs = [unitary_to_statevector(unitary) for unitary in evol]
        return [(statevec.conj().T@measurement)@statevec for statevec in statevecs], depth 



def U_trotter(init, n, V, coeff, t_max, n_qubits,state, order=1, protect=False, rand=False, measurement=None,error=None):
    evol = []
    evol.append(init)
    depth = [0]
    t_step = t_max/n
    m = init
    V2 = V + V[::-1]
    coeff2 = (coeff + coeff[::-1])
    if rand:
        V = np.random.permutation(V)
    for i in range(1,n+1):
        if order==1:
            if error != None:
                if protect:
                    angles = [-0.01*np.pi*n for _ in range(n_qubits)]
                    protection = add_depolarizing_error(tensor_product_rz(angles),n_qubits,error)
                    m = protection@m
                for j in range(len(V)):
                    result_matrix =  apply_depolarizing_error(pauli_matrices[V[j][0]],error)
                    for op in V[j][1:]:
                        result_matrix = np.kron(result_matrix, apply_depolarizing_error(pauli_matrices[op],error))
                    identity = np.eye(init.shape[0],init.shape[1])
                    m = (np.cos(coeff[j] * t_step)*identity - 1j*np.sin(coeff[j] * t_step)*result_matrix )@m
                if protect:
                    angles = [0.01*np.pi*n for _ in range(n_qubits)]
                    protection = add_depolarizing_error(tensor_product_rz(angles),n_qubits,error)
                    m = protection@m
            else:
                if protect:
                    angles = [-np.pi*n for _ in range(n_qubits)]
                    protection = tensor_product_rz(angles)
                    m = protection@m
                for j in range(len(V)):
                    result_matrix =  pauli_matrices[V[j][0]] 
                    for op in V[j][1:]:
                        result_matrix = np.kron(result_matrix, pauli_matrices[op])
                    identity = np.eye(init.shape[0],init.shape[1])
                    m = (np.cos(coeff[j] * t_step)*identity - 1j*np.sin(coeff[j] * t_step)*result_matrix )@m
                if protect:
                    angles = [np.pi*n for _ in range(n_qubits)]
                    protection = tensor_product_rz(angles)
                    m = protection@m
            depth.append(depth[-1]+len(V))
        elif order==2:
            if error!=None:
                if protect:
                    angles = [-0.01*np.pi*n for _ in range(n_qubits)]
                    protection = add_depolarizing_error(tensor_product_rz(angles),n_qubits,error)
                    m = protection@m
                for j in range(len(V2)):
                    result_matrix =  apply_depolarizing_error(pauli_matrices[V2[j][0]],error)
                    for op in V2[j][1:]:
                        result_matrix = np.kron(result_matrix, apply_depolarizing_error(pauli_matrices[op],error))
                    identity = np.eye(init.shape[0],init.shape[1])
                    m = (np.cos(coeff2[j]/2 * t_step)*identity - 1j*np.sin(coeff2[j]/2 * t_step)*result_matrix )@m     
                if protect:
                    angles = [0.01*np.pi*n for _ in range(n_qubits)]
                    protection = add_depolarizing_error(tensor_product_rz(angles),n_qubits,error)
                    m = protection@m 
            else:
                if protect:
                    angles = [-0.8*np.pi*n for _ in range(n_qubits)]
                    protection = tensor_product_rz(angles)
                    m = protection@m
                for j in range(len(V2)):
                    result_matrix =  pauli_matrices[V2[j][0]] 
                    for op in V2[j][1:]:
                        result_matrix = np.kron(result_matrix, pauli_matrices[op])
                    identity = np.eye(init.shape[0],init.shape[1])
                    m = (np.cos(coeff2[j]/2  * t_step)*identity - 1j*np.sin(coeff2[j]/2 * t_step)*result_matrix )@m 
                if protect:
                    angles = [0.8*np.pi*n for _ in range(n_qubits)]
                    protection = tensor_product_rz(angles)
                    m = protection@m
            depth.append(depth[-1]+len(V2))
        evol.append(m)
    if measurement==None:
        return evol, depth
    elif isinstance(measurement,str):
        statevecs = [unitary_to_statevector(unitary) for unitary in evol]
        Z = sum([ft.reduce(np.kron,[np.array([[1, 0], [0, 1]])]*n_qubits) for _ in range(6)]) - sum([pauli_string_to_matrix(replacer('I'*n_qubits, idx, 'Z')) for idx in range(6)])
        return [(statevec.conj().T@Z)@statevec for statevec in statevecs], depth
    else:
        statevecs =[unitary_to_statevector(unitary) for unitary in evol]
        return [(statevec.conj().T@measurement)@statevec for statevec in statevecs], depth 
