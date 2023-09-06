from qDrift.hamsimqDrift import AlgorithmHamSimqDrift
from pytket.circuit import Circuit
from openfermion.utils.operator_utils import count_qubits

from utils.func import *
from utils.term_grouping import *
from utils.min_clique import *
from utils.hamlib import *
from utils.evol_real import *
from utils.plot import *
from utils.driver import *
from utils.JW import *

from Pauli_Gadgets.paulis import *

import functools as ft
import numpy as np
import matplotlib.pyplot as plt

def gen_sys_param(n, sample_steps=100000, N=1):
    global t_max
    global initial_state_circ
    global n_qdrift_steps
    global sample_space
    global seg
    global t_list
    global error_threshold

    t_list = [0.5,1,2,5]
    t_max = t_list[-1]
    seg = N
    error_threshold = 1e-3
    initial_state_circ = Circuit(n)
    for i in range(n):
        initial_state_circ.H(i)
    n_qdrift_steps = sample_steps
    sample_space = np.arange(0,sample_steps+1,1)

def Hchain(n=3):
    driver = PySCFDriver_custom()
    bond_length = 1.0
    n_sites = n
    driver.load_molecule(
        atom=[('H', (0, 0, i * bond_length)) for i in range(n_sites)], # Create a molecular data object for the hydrogen chain
        basis="sto-3g",          # Basis set for quantum chemistry calculation
        multiplicity=n_sites%2,          # Spin multiplicity for molecule, since the total spin of H2O is S=0，its spin multiplicity is 2S+1=1
        charge=0,                 # Total charge of molecule, since H2O is charge neutral, its charge=0
        unit="Angstrom"
    )
    driver.run_scf()             # Perform Hartree Fock calculation

    # np.set_printoptions(precision=4, linewidth=150)

    hpq = driver.get_onebody_tensor("int1e_kin") + driver.get_onebody_tensor("int1e_nuc")
    vpqrs = driver.get_twobody_tensor()
    # assert np.shape(hpq)==(7, 7)             # H2O has 7 orbitals when using STO-3G basis.
    # assert np.shape(vpqrs)==(7, 7, 7, 7)

    # print(hpq)
    operator = get_molecular_hamiltonian(hpq,vpqrs,driver)
    n_qubits = count_qubits(operator)
    number, coulomb, hopping, no_excitation, double_excitation = JW_transformation(operator)
    numbers, coulombs, hoppings, no_excitations, double_excitations = JW_transformation(operator,True)

    number_op, number_co = convert_op_to_input(number,n_qubits)
    hopping_op, hopping_co = convert_op_to_input(hopping,n_qubits)
    coulomb_op, coulomb_co = convert_op_to_input(coulomb,n_qubits)
    no_excitation_op, no_excitation_co = convert_op_to_input(no_excitation,n_qubits)
    double_excitation_op, double_excitation_co = convert_op_to_input(double_excitation,n_qubits)

    numbers_op, numbers_co = convert_op_to_input(numbers,n_qubits,True)
    hoppings_op, hoppings_co = convert_twobody_op_to_input(hoppings,n_qubits,True)
    coulombs_op, coulombs_co = convert_op_to_input(list(coulombs),n_qubits,True)
    no_excitations_op, no_excitations_co = convert_twobody_op_to_input(list(no_excitations),n_qubits,True)
    double_excitations_op, double_excitations_co = convert_twobody_op_to_input(list(double_excitations),n_qubits,True)

    max_part_group, max_coeff = [numbers_op+hoppings_op+coulombs_op+no_excitations_op+double_excitations_op], [numbers_co+hoppings_co+coulombs_co+no_excitations_co+double_excitations_co]

    H_matrix = get_Hmatrix(number_op+hopping_op+coulomb_op+no_excitation_op+double_excitation_op,n_qubits,number_co+hopping_co+coulomb_co+no_excitation_co+double_excitation_co)
    return max_part_group[0],max_coeff[0],H_matrix,[[h] for h in number_op+hopping_op+coulomb_op+no_excitation_op+double_excitation_op],[[c] for c in number_co+hopping_co+coulomb_co+no_excitation_co+double_excitation_co],n_qubits

physDrift = []
qDrift = []

for sysSize in np.arange(3,4):
    for N in [100]:
        phyV,phyC,H,qV,qC,n = Hchain(sysSize)
        secdepth = np.arange(0,(N+1)*142,142)

        gen_sys_param(n=n,sample_steps=secdepth[-1])

        drift_time_evolution = AlgorithmHamSimqDrift(initial_state_circ,phyV,phyC,t_max,n_qdrift_steps,seg,M=3,noise=True)
        Um,sm,mdepth = drift_time_evolution.Drift_exp(depth=secdepth[-1], cheat=True)
        Um= [[Um[j][0][i] for i in range(len(Um[j][0]))] for j in range(len(Um))]

        drift_time_evolution_parity = AlgorithmHamSimqDrift(initial_state_circ,qV,qC,t_max,n_qdrift_steps,seg,M=3,noise=True)
        Uq,sq,qdepth = drift_time_evolution_parity.Drift_exp(depth=secdepth[-1], cheat=True)
        Uq= [[Uq[j][0][i] for i in range(len(Uq[j][0]))] for j in range(len(Uq))]

        Uexc = U_exc(drift_time_evolution.circuit.get_unitary(),n_qdrift_steps,t_max, H)
        uexc = extract_U_at_t(t_list,Uexc,secdepth)

        um = [extract_U_at_t(t_list,Um[i],secdepth) for i in range(len(Um))]
        um_spec = [[np.abs(linalg.eig(u - u_exc)[0]).max() for u,u_exc in zip(um[i],uexc)] for i in range(len(um))]
        uq = [extract_U_at_t(t_list,Uq[i],secdepth) for i in range(len(Uq))]
        uq_spec = [[np.abs(linalg.eig(u - u_exc)[0]).max() for u,u_exc in zip(uq[i],uexc)] for i in range(len(uq))]

        physDrift.append(np.mean(um_spec,axis=0))
        qDrift.append(np.mean(uq_spec,axis=0))
        print(np.mean(um_spec,axis=0),np.mean(uq_spec,axis=0))
# [0.         0.00300629 0.00358299 0.00510126 0.0076013 ] [0.         0.00517473 0.00565317 0.00696613 0.00820971]

f = open('output.txt',"w+")
for i in range(len(physDrift)):
    f.write(str([physDrift[i],qDrift[i]]))
f.close()