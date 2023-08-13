from openfermion import QubitOperator as q
from openfermion import get_sparse_operator
from copy import deepcopy
from utils.min_clique import BronKerbosch

# Get ops in cirq format
def get_openfermion_str(pauli_term, reverse=True):
    
    cirq_term = []
    
    for i, op in enumerate(list(pauli_term)):
        if op == 'I':
            continue
        
        cirq_term.append(op + str(9-i if not reverse else i))
    new_pauli_term = ' '.join(cirq_term)
    
    return new_pauli_term

def get_dict_str(pauli_term, reverse=True):
    
    cirq_term = []
    
    for i, op in enumerate(list(pauli_term)):
        if op == 'I':
            continue
        
        cirq_term.append(op + str(9-i if not reverse else i))
    # removed space between terms to make hist plot compact
    new_pauli_term = ''.join(cirq_term)
    
    return new_pauli_term

def ops_commute(op1, op2):
    sign = 1
    for pauli_1, pauli_2 in zip(list(op1), list(op2)):
        if pauli_1=='I' or pauli_2=='I' or pauli_1==pauli_2:
            continue
        sign *= -1
    
    return True if sign==1 else False

def ops_do_not_overlap(op_1, op_2):
    qbs = []
    for i, (p1, p2) in enumerate(zip(list(op_1), list(op_2))):
        # print(p1,p2)
        if p1!='I' and p2!='I':
            return False
    return True

def match_weight(op_1, op_2):
    weight = 0.
    for p1, p2 in zip(list(op_1), list(op_2)):
        if p1=='I' and p2=='I':
            continue
        elif p1=='I' or p2=='I':
            weight += 1.
        elif p1 == p2:
            weight += 0.
        else:
            weight += 2.
    return weight

# From https://stackoverflow.com/a/10824420
def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            yield from flatten(i)
        else:
            yield i

def generate_dict(H):
    # Get ops and coeffs as list
    split_input = H.split()
    coeffs = []
    ops = []
    for i, term in enumerate(split_input):
        if '0' in term:
            coeff = float(term)
            sign = 1 if split_input[i-1]=='+' else -1
            coeffs.append(sign * coeff)
        elif term[0] in ['I','X','Y','Z']:
            ops.append(term)

    n_qubits = len(ops[0])
    # Get Openfermion hamiltonian
    H = q()

    cirq_ops = []
    for op in ops:
        cirq_ops.append(get_openfermion_str(op))

    for coeff, op in zip(coeffs, cirq_ops):
        H += q(op, coeff)

    H_matrix = get_sparse_operator(H)

    # Dictionary with coeffs
    ops_dict = {}
    for coeff, op in zip(coeffs, ops):
        ops_dict[op] = coeff

    #Using reduced coefficients
    red_coeffs = []
    red_ops = []
    for coeff, op in zip(coeffs, ops):
        if abs(coeff) > 0.0035: #0.0035
            red_coeffs.append(coeff)
            red_ops.append(op)
            
    # Terms associated with each coeff value
    coeff_groups = {}
    #for coeff, op in zip(coeffs, ops):
    for coeff, op in zip(red_coeffs, red_ops):
        if coeff not in coeff_groups:
            coeff_groups[coeff] = [op]
        else:
            coeff_groups[coeff].append(op)
    # print(coeff_groups)
    # Check whether each set of terms for a coeff value has full commutativity
    coeff_commuting_groups = {}
    for coeff, op_list in coeff_groups.items():
        com = True
        for op1 in op_list:
            for op2 in op_list:
                com = com and ops_commute(op1, op2)
        coeff_commuting_groups[coeff] = com

    # print(coeff_commuting_groups)
    coeff_groups_sorted = dict(sorted(coeff_groups.items(), key = lambda x: -abs(x[0])))

    sorted_ops = []
    for coeff, op_list in coeff_groups_sorted.items():
        for op in op_list:
            sorted_ops.append(op)

    ops_no_I = deepcopy(sorted_ops)
    identity = 'I'*n_qubits
    ops_no_I.remove(identity)

    H_ops_orig = [[op] for op in ops]
    H_coeff_orig = [[coeff] for coeff in coeffs]

    return H_matrix, ops_dict, ops_no_I, H_ops_orig, H_coeff_orig


def create_clique(ops2, ops_dict):
    # We start by implementing the first step of 2001.05983,
    # which solves a minimum-clique-cover problem to find
    # a term grouping such that each group's terms mutually
    # commute, and the number of terms is minimized.
    # This is solved by using the Bron-Kerbosh algorithm.

    comm_dict = {}
    for op_1 in ops2:
        comm_list = []
        for op_2 in ops2:
            if op_1 == op_2:
                continue   
            if ops_commute(op_1, op_2):
                comm_list.append(op_2)
        comm_dict[op_1] = comm_list

    min_clique_cover_orig = BronKerbosch(comm_dict)
    # We reorder the terms within each group, we try to keep the performance
    # near the original sorted order, which we already low has low error.

    min_clique_cover = []
    for clique in min_clique_cover_orig:
        lclique = list(clique)
        min_clique_cover.append(lclique)
    # print(min_clique_cover)
    min_clique_cover = [sorted(clique, key=lambda op: abs(ops_dict[op]), reverse=True) for clique in min_clique_cover]
    # Get optimal ordering
    return min_clique_cover