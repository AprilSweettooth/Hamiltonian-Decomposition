import argparse
from ham_lib import *
import saveload_utils as slu

MOLECULES = ["h2", "lih", "beh2", "h2o", "nh3", "n2"]
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--molecule", type=str, default="h2", required=False, choices=MOLECULES
)
args = parser.parse_args()

Hq = slu.load_qubit_hamiltonian(mol=args.molecule)
print(f"Hamiltonian: {Hq}")
