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

def convert_op_to_input(ops,n):
    tk_op = []
    tk_coeff = []
    for term, coeff in ops.terms.items():
        tk_op.append(replace_Pauli_strings(n,list(term)))
        tk_coeff.append(coeff)
    return tk_op[1:],tk_coeff[1:]

def convert_twobody_op_to_input(ops,n):
    tk_op = []
    tk_coeff = []
    for term, coeff in ops.terms.items():
        tk_op.append(replace_Pauli_strings(n,list(term)))
        tk_coeff.append(coeff)
    return tk_op,tk_coeff

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

