"""The file contains method that save & load Hamiltonians
"""
import pickle
from ham_lib import *
from openfermion import bravyi_kitaev

def load_fermionic_hamiltonian(mol, prefix="./"):
    with open(prefix + "ham_lib/" + mol + "_fer.bin", "rb") as f:
        Hf = pickle.load(f)
    return Hf


def load_interaction_hamiltonian(mol, prefix="./"):
    with open(prefix + "ham_lib/" + mol + "_int.bin", "rb") as f:
        H = pickle.load(f)
    return H


def load_qubit_hamiltonian(mol, tf="bk", prefix="/Users/peteryang/Downloads/Hamiltonian-Decomposition/VQE_Hamiltonian-main/"):
    with open(prefix + "ham_lib/" + mol + "_fer.bin", "rb") as f:
        Hf = pickle.load(f)
    if tf == "bk":
        return bravyi_kitaev(Hf)
    else:
        raise ValueError(f"Transformation {tf} not supported")
