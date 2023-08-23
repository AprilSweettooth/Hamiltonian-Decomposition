from numpy.linalg import matrix_power
import scipy.sparse.linalg as ssl
import numpy as np
from scipy.linalg import expm
from openfermion import QubitOperator as q
from openfermion import get_sparse_operator
from utils.term_grouping import get_openfermion_str

def get_Hmatrix(V,n,co):
    Hq = q()

    cirq_ops = []
    for op in V:
        cirq_ops.append(get_openfermion_str(op,n))

    for co, op in zip(co, cirq_ops):
        Hq += q(op, co)

    H_matrix = get_sparse_operator(Hq)
    return H_matrix

def U_exc(init, n, t_max, H_matrix):
    t_step = t_max/n
    U_mat = ssl.expm(-1j * H_matrix * t_step).toarray()
    # print(U_mat.shape)
    evol = [ matrix_power(U_mat,i)@init for i in range(n+1) ]
    return evol


# Define the Pauli operator matrices
pauli_matrices = {
    'X': np.array([[0, 1], [1, 0]]),
    'Y': np.array([[0, -1j], [1j, 0]]),
    'Z': np.array([[1, 0], [0, -1]]),
    'I': np.array([[1, 0], [0, 1]])
}

def pauli_string_to_matrix(pauli_string, coeff):
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

def U_drift(init, rep, n, t_max, V, coeff):
    # evol = []
    # evol.append(init)
    # t_step = t_max/n
    # m = init
    # for _ in range(1,n+1):
    #     for i in range(len(coeff)):
    #         for v in V[i]:
    #             # Hq = q()
    #             # print(get_openfermion_str([*v],n=len([*v])))
    #             # Hq += q(get_openfermion_allstr([*v],n=len([*v])), coeff[i])
    #             H_matrix = pauli_string_to_matrix(v, coeff[i]*np.pi/2)
    #             # print(H_matrix)
    #             # print(H_matrix.shape)
    #             m = expm(-1j * H_matrix * t_step)@m
    #     # print(m)
    #     evol.append(m)
    # return evol
    evol = []
    evol.append(init)
    t_step = t_max/rep
    m = init
    for j in range(1,n+1):
        for i in range(rep):
            # print((j-1)*rep+i)
            for v in V[(j-1)*rep+i]:
                result_matrix =  pauli_matrices[v[0]] 
                for op in v[1:]:
                    result_matrix = np.kron(result_matrix, pauli_matrices[op])
                identity = np.eye(init.shape[0],init.shape[1])
                m = (np.cos(coeff[(j-1)*rep+i] * np.pi/4 * t_step)*identity - 1j*np.sin(coeff[(j-1)*rep+i] * np.pi/4 * t_step)*result_matrix )@m
        # print(m)
        evol.append(m)
    return evol


