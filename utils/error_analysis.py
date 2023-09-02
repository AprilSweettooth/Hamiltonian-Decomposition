import numpy as np
from scipy import linalg

from utils.func import search_U_with_no_exp, extract_U_at_t

def extract_depth_at_t(t,depth):
    t_step = t[-1]/depth[0][-1]
    depth_new = [[depth[i][0]] for i in range(len(depth))]
    for i in range(len(depth)):
        for j in range(len(t)):
            idx = int(t[j] / t_step)
            depth_new[i].append(depth[i][search_U_with_no_exp(idx,depth[i])])
    mean, std = np.mean(depth_new,axis=0), np.std(depth_new,axis=0)
    xerror = [(s/(m+0.0001))*t_step for s,m in zip(std,mean)]
    return xerror

def cal_std_drift(U, Ur, t, depth, secdepth, t_list, Uexc, ave=False):
    u_noise = []
    t_step = t[-1]/depth[0][-1]
    depth_new = [[depth[i][0]] for i in range(len(depth))]
    for i in range(len(depth)):
        for j in range(len(t)):
            idx = int(t[j] / t_step)
            depth_new[i].append(depth[i][search_U_with_no_exp(idx,depth[i])])
    if not ave:
        for i in range(len(U)):
            u_t = extract_U_at_t(t_list,U[i],secdepth)
            Urt = [Ur[depth_new[i][j]] for j in range(len(depth_new[i]))]
            u_noise.append([np.abs(linalg.eig(u - u_exc)[0]).max() for u,u_exc in zip(u_t,Urt)])
        # print(u_noise)
        u_noise_mean, u_noise_std = np.mean(u_noise,axis=0),np.std(u_noise,axis=0) 
        return u_noise_mean, u_noise_std 
    else:
        u_t = []
        for i in range(len(U)):
            u_t.append(extract_U_at_t(t_list,U[i],secdepth))
        # print(u_t[0][0])
        Urt = extract_U_at_t(t_list,Uexc,np.arange(secdepth[-1]))
        # print(np.abs(linalg.eig(np.add(np.add(Um[0][0],Um[1][0]),Um[2][0])/3-Urt[0])[0]).max())
        u = [np.add(np.add(np.array(u_t[0][i]),np.array(u_t[1][i])),np.array(u_t[2][i]))/3 for i in range(len(u_t[0]))]
        # print(u[0])
        error = [np.abs(linalg.eig(u - u_exc)[0]).max() for u,u_exc in zip(u,Urt)]
        # print(error)
        return error