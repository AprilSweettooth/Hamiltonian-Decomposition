import numpy as np
from scipy import linalg

def factorial(n):
    if n < 2:
        return 1
    else:
        return n * factorial(n-1)
    
# Calculate the L1 spectral distance between the two unitary matrices which is the operator norm, largest eigen (singular) value
def calculate_error(U1, U2):
    # U = np.zeros((U1[0].shape), dtype='complex128')
    M = len(U1)
    U_sim = sum(U1)
    # print(U_sim/M)
    # print(U2)
    # print(U_sim/M - U2) 
    return np.abs(linalg.eig(U_sim/M - U2)[0]).max()

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