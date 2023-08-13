from numpy.linalg import matrix_power
import scipy.sparse.linalg as ssl

def U_exc(init, n, t_max, H_matrix):
    t_step = t_max/n
    U_mat = ssl.expm(-1j * H_matrix * t_step).toarray()
    evol = [ matrix_power(U_mat,i)@init for i in range(n+1) ]
    return evol