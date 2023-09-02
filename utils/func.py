import numpy as np
from scipy import linalg
from typing import Tuple, Optional
from itertools import product
from openfermion import InteractionOperator
from pytket.pauli import Pauli, QubitPauliString
from pytket.circuit import Qubit
from pytket.utils import QubitPauliOperator

def factorial(n):
    if n < 2:
        return 1
    else:
        return n * factorial(n-1)
    
# Calculate the L1 spectral distance between the two unitary matrices which is the operator norm, largest eigen (singular) value
def calculate_error(U1, U2):
    # U = np.zeros((U1[0].shape), dtype='complex128')
    # M = len(U1)
    # U_sim = sum(U1)
    # print(U_sim/M)
    # print(U2)
    # print(U_sim/M - U2) 
    return np.abs(linalg.eig(U1 - U2)[0]).max()

def count_parity(dict):
    parity = [i for i in list(list(dict[0].keys())[0])].count(1)
    ratio = np.zeros((len(dict),))
    ratio[0] = 1
    for i in range(1,len(dict)):
        for j in list(dict[i]):
            if list(list(j)).count(1) == parity:
                ratio[i] += dict[i][j]
        ratio[i] = ratio[i]/sum(list(dict[i].values()))
    return ratio

def qps_from_openfermion(paulis):
    """Convert OpenFermion tensor of Paulis to pytket QubitPauliString."""
    pauli_sym = {"I": Pauli.I, "X": Pauli.X, "Y": Pauli.Y, "Z": Pauli.Z}
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

def strings_from_openfermion(openf_op):
    """Convert OpenFermion QubitOperator to pytket QubitPauliOperator."""
    tk_op = dict()
    for term, coeff in openf_op.terms.items():
        string = qps_from_openfermion(term)
        tk_op[string] = coeff
    return tk_op

def replace_Pauli_strings(n,list_paulis):
    identity = 'I'*n
    for p in list_paulis:
        pos = list(p)[0]
        pauli = list(p)[1]
        identity = identity[:pos] + pauli + identity[pos + 1:] 
    return identity

def convert_op_to_input(ops,n,nested=False):
    tk_op = []
    tk_coeff = []
    if nested:
        for i in range(len(ops)):
            tk_op.append([])
            tk_coeff.append([])
            for term, coeff in ops[i].terms.items():
                tk_op[i].append(replace_Pauli_strings(n,list(term)))
                tk_coeff[i].append(coeff)
        return [op[1:] for op in tk_op], [co[1:] for co in tk_coeff]
    else:
        for term, coeff in ops.terms.items():
            tk_op.append(replace_Pauli_strings(n,list(term)))
            tk_coeff.append(coeff)
        return tk_op[1:],tk_coeff[1:]

def convert_twobody_op_to_input(ops,n,nested=False):
    tk_op = []
    tk_coeff = []
    if nested:
        for i in range(len(ops)):
            tk_op.append([])
            tk_coeff.append([])
            for term, coeff in ops[i].terms.items():
                tk_op[i].append(replace_Pauli_strings(n,list(term)))
                tk_coeff[i].append(coeff)
        return tk_op, tk_coeff
    else:
        for term, coeff in ops.terms.items():
            tk_op.append(replace_Pauli_strings(n,list(term)))
            tk_coeff.append(coeff)
        return tk_op,tk_coeff

def search_depth_with_no_exp(idx,depth):
    for i in range(1,len(depth)):
        # print(depth[i])
        if idx > depth[i]:
            continue
        elif idx == depth[i]:
            return depth[i]
        elif idx < depth[i]:
            if depth[i]-idx > idx-depth[i-1]:
                return depth[i]
            else:
                return depth[i-1]

def perm_ops(t,depth,sm):
    t_step = t[-1]/depth[-1]
    ind = [[],[],[]]
    V = [[],[],[]]
    coeff = [[],[],[]]
    depth_new = [[depth[i][0]] for i in range(len(depth))]
    for i in range(len(depth)):
        for j in range(len(t)):
            idx = int(t[j] / t_step)
            depth_new[i].append(search_depth_with_no_exp(idx,depth[i]))
    # print(depth_new)
    for i in range(len(depth_new)):
        for j in range(len(depth_new[i])):
            ind[i].append(depth[i].index(depth_new[i][j]))
    for i in range(len(ind)):
        for j in range(len(ind[i])-1):
            V[i].append(np.random.permutation(sm[0][i][ind[i][j]:ind[i][j+1]]))
            coeff[i].append(np.random.permutation(sm[1][i][ind[i][j]:ind[i][j+1]]))
    return [[item for row in V[i] for item in row] for i in range(3)],[[item for row in coeff[i] for item in row] for i in range(3)]

def search_U_with_no_exp(idx,depth):
    count = 0
    idx_tmp = idx
    while idx_tmp>0:
        idx_tmp -= depth[count]
        count += 1
    count -= 1
    if abs(idx_tmp) > depth[count]//2:
        return count - 1
    else:
        return count
def extract_U_at_t(t,U,depth):
    if t[-1]/depth[-1] > t[0]:
        raise Exception('Get a smaller sample steps')
    U_new = [U[0]]
    t_step = t[-1]/depth[-1]
    for i in range(len(t)):
        idx = int(t[i] / t_step)
        U_new.append(U[search_U_with_no_exp(idx,depth)])
    return U_new

def Monte_Carlo_ave(t,U,depth,Ur,M=3):
    U_new = []
    U_mean = []
    for i in range(M):
        U_new.append(extract_U_at_t(t,U[i],depth[i]))
    for j in range(M):
        U_mean.append([np.abs(linalg.eig(u - u_exc)[0]).max() for u,u_exc in zip(U_new[j],Ur)])
    return np.mean(U_mean,axis=0),np.std(U_mean,axis=0) 

def site_excitation_group(number,hopping,nco,hco):
    group = []
    coeff = []
    for i in range(len(hopping)):
        group.append([])
        coeff.append([])
        pos = [idx for idx in range(len(hopping[i][0])) if hopping[i][0][idx]=='X' ]
        if len(pos) != 2:
            raise Exception('Excitation not conserved')
        for j in range(2):
            group[i].append(number[pos[j]][0])
            coeff[i].append(nco[pos[j]][0])
            number[pos[j]] = 'I'*len(number[0])
            nco[pos[j]] = 0 
        group[i].append(hopping[i][0])
        group[i].append(hopping[i][1])
        coeff[i].append(hco[i][0])
        coeff[i].append(hco[i][1])
    group.append([])
    coeff.append([])
    for k in range(len(number)):
        if number[k][0] != 'I'*len(number[0]):
            group[-1].append(number[k][0])
            coeff[-1].append(nco[k][0])
    return group, coeff

# !/usr/bin/env python3
# Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def orb2spinorb(
    num_modes: int,
    single_ex_amps: Optional[np.ndarray] = None,
    double_ex_amps: Optional[np.ndarray] = None
) -> Tuple[np.ndarray]:
    r"""
    Transform molecular orbital integral into spin orbital integral, assume
    the quantum system is spin restricted.

    Args:
        num_modes: Number of molecular orbitals.
        single_ex_amps: One electron integral.
        double_ex_amps: Two electron integral.
    
    Return:
        The molecular integral in spin orbital form.
    """
    if isinstance(single_ex_amps, np.ndarray):
        assert single_ex_amps.shape == (num_modes, num_modes)
        single_ex_amps_so = np.zeros((2*num_modes, 2*num_modes))
        for p in range(num_modes):
            single_ex_amps_so[2*p, 2*p] = single_ex_amps[p, p]
            single_ex_amps_so[2*p+1, 2*p+1] = single_ex_amps[p, p]
            for q in range(p+1, num_modes):
                single_ex_amps_so[2*p, 2*q] = single_ex_amps[p, q]
                single_ex_amps_so[2*p+1, 2*q+1] = single_ex_amps[p, q]

                single_ex_amps_so[2*q, 2*p] = single_ex_amps[p, q]
                single_ex_amps_so[2*q+1, 2*p+1] = single_ex_amps[p, q]
    if isinstance(double_ex_amps, np.ndarray):
        assert double_ex_amps.shape == (num_modes, num_modes, num_modes, num_modes)
        double_ex_amps_so = np.zeros((2*num_modes, 2*num_modes, 2*num_modes, 2*num_modes))
        for p, r, s, q in product(range(num_modes), repeat=4):
            double_ex_amps_so[2*p, 2*r, 2*s, 2*q] = double_ex_amps[p, r, s, q]
            double_ex_amps_so[2*p+1, 2*r, 2*s, 2*q+1] = double_ex_amps[p, r, s, q]
            double_ex_amps_so[2*p, 2*r+1, 2*s+1, 2*q] = double_ex_amps[p, r, s, q]
            double_ex_amps_so[2*p+1, 2*r+1, 2*s+1, 2*q+1] = double_ex_amps[p, r, s, q]
    
    if isinstance(single_ex_amps, np.ndarray) and isinstance(double_ex_amps, np.ndarray):
        return single_ex_amps_so, double_ex_amps_so
    elif isinstance(single_ex_amps, np.ndarray):
        return single_ex_amps_so
    elif isinstance(double_ex_amps, np.ndarray):
        return double_ex_amps_so
    else:
        raise ValueError("One of the `single_ex_amps` and `double_ex_amps` should be an np.ndarray.")
    
def get_molecular_hamiltonian(hpq, vpqrs, driver):
    r"""returns the molecular hamiltonian for the given molecule.
    """
    constant = driver.energy_nuc

    hcore_so, eri_so = orb2spinorb(driver.num_modes, hpq, vpqrs)

    # set the values in the array that are lower than 1e-16 to zero.
    eps = np.finfo(hpq.dtype).eps
    hcore_so[abs(hcore_so) < eps] = 0.0
    eri_so[abs(eri_so) < eps] = 0.0

    h = InteractionOperator(constant, hcore_so, 0.5*eri_so)
    return h

