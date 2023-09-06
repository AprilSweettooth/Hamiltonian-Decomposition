import matplotlib.pyplot as plt
from scipy import linalg
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from collections import Counter
# import pandas as pd

def cheat_plot(measurements, labels, ylabel, color):
    times = list(measurements.keys())
    exps = list(measurements.values())
    plt.plot(times, exps, label=labels, c=color)
    plt.xlabel(r'Time')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def evol_plot(real_measu, times, exps, gate, infidelity, labels, color):
    real_exps = list(real_measu.values())
    fig, axs = plt.subplots(1, 3, sharex='col', figsize=(3*6, 4))
    axs[0].set_title('Projetion on initial state')
    axs[0].set_ylim([-0.05,1.05])
    axs[0].set_xlabel(r'Time')
    axs[0].scatter(times, exps, label=labels[0], c=color)
    axs[0].plot(times, real_exps, label='real Hamiltonian', c='blue')
    axs[1].set_title('CX gate count')
    axs[1].set_xlabel(r'Time')
    axs[1].plot(times, gate, label=labels[1])
    axs[2].set_title('infidelity evolution')
    axs[2].set_xlabel(r'Time')
    axs[2].plot(times, infidelity, label=labels[2])
    for i in range(3):
        axs[i].legend() 
    plt.show()

def compare_plot(real_measu, times, exps, gates, infidelities, labels, color):
    real_exps = list(real_measu.values())
    fig, axs = plt.subplots(1, 3, sharex='col', figsize=(3*6, 4))
    axs[0].plot(times, real_exps, label='real Hamiltonian')
    for i in range(len(exps)):
        axs[0].scatter(times, exps[i], label=labels[i], c=color[i])
        axs[1].plot(times, gates[i], label=labels[i], c=color[i])
        axs[2].plot(times, infidelities[i], label=labels[i], c=color[i])
    axs[0].set_xlabel(r'Time')
    axs[0].set_ylabel(r'Prob')
    axs[0].set_ylim([-0.05,1.05])
    axs[1].set_xlabel(r'Time')
    axs[1].set_ylabel(r'# cx-gates')
    axs[2].set_xlabel(r'Time')
    axs[2].set_ylabel(r'# infidelity')
    for i in range(3):
        axs[i].legend()
    plt.show()

def exp_plot(real_measu, measurements, labels, exp, color):
    times = list(measurements.keys())
    real_exps = list(real_measu.values())
    exps = list(measurements.values())
    plt.plot(times, exps, label=labels, c=color)
    plt.plot(times, real_exps, label='real Hamiltonian', c='blue')
    plt.xlabel(r'Time')
    # plt.ylim([-2.5,-1.5])
    plt.ylabel(exp)
    plt.legend()
    plt.show()

def particle_number(sample, shots_ratio, label, color):
    plt.plot(sample,shots_ratio,label=label, c=color)
    plt.legend()
    plt.title('Particle number evolution')
    plt.xlabel('Sample number')
    plt.ylabel('scaled particle number')

def compare_particle_number(sample, shots_ratio):
    plt.plot(sample,shots_ratio[0],label='term grouped')
    plt.plot(sample,shots_ratio[1],label='original qdrift')
    plt.title('Particle number evolution')
    plt.xlabel('Sample number')
    plt.ylabel('scaled particle number')
    plt.legend()

def take_id(U,sample):
    U_new = []
    for i in range(len(sample)):
        U_new.append(U[sample[i]])
    return U_new

# def compare_spectral_error(sample_number, U, Us, Us_orig, labels, grouped=None, Us_prot=None, sample=None):
#     if isinstance(grouped,list):
#         plt.plot(sample_number[0::grouped[0]],[np.abs(linalg.eig(u - u_exc)[0]).max() for u,u_exc in zip(Us,U[0::grouped[0]])], label=labels[0])
#         plt.plot(sample_number[0::grouped[1]],[np.abs(linalg.eig(u - u_exc)[0]).max() for u,u_exc in zip(Us_orig,U[0::grouped[1]])], label=labels[1])
#     elif grouped != None: 
#         plt.plot([0]+[len(s) for s in sample],[np.abs(linalg.eig(u - u_exc)[0]).max() for u,u_exc in zip(Us,take_idx(U,sample))], label=labels[0])
#         plt.plot(sample_number,[np.abs(linalg.eig(u - u_exc)[0]).max() for u,u_exc in zip(Us_orig,U)], label=labels[1])
#         if Us_prot != None:
#             plt.plot([0]+[len(s) for s in sample],[np.abs(linalg.eig(u - u_exc)[0]).max() for u,u_exc in zip(Us_prot,take_idx(U,sample))], label=labels[2]) 
#     else:
#         plt.plot(sample_number,[np.abs(linalg.eig(u - u_exc)[0]).max() for u,u_exc in zip(Us,U)], label=labels[0])
#         plt.plot(sample_number,[np.abs(linalg.eig(u - u_exc)[0]).max() for u,u_exc in zip(Us_orig,U)], label=labels[1])        
#     plt.title('spectral error over sample number')
#     plt.xlabel('Sample steps')
#     plt.ylabel('spectral error') 
#     plt.legend()

def compare_spectral_error(depth,U,Uexc,labels):
    for i in range(len(depth)):
        plt.plot(depth[i],[np.abs(linalg.eig(u - u_exc)[0]).max() for u,u_exc in zip(U[i],take_id(Uexc,depth[i]))],label=labels[i])
    plt.title('spectral error over circuit depth')
    plt.xlabel('number of pauli gadgets')
    plt.ylabel('spectral error') 
    plt.legend()

def hist(dicts):
    # Creating histogram
    fig, axs = plt.subplots(1, 2,
                            figsize =(2*6, 7),
                            tight_layout = True)
    # mask11 = list(dicts[0].values()) < max(list(dicts[0].values())) * 0.25
    # mask12 = max(list(dicts[0].values())) * 0.5 <= list(dicts[0].values()) < max(list(dicts[0].values())) * 0.5
    # mask13 = max(list(dicts[0].values())) * 0.5 <= list(dicts[0].values())
    # mask21 = list(dicts[1].values()) < max(list(dicts[1].values())) * 0.25
    # mask22 = max(list(dicts[1].values())) * 0.5 <= list(dicts[1].values()) < max(list(dicts[1].values())) * 0.5
    # mask23 = max(list(dicts[1].values())) * 0.5 <= list(dicts[1].values() ) 
    # fracs = ((N**(1 / 5)) / N.max())
    # norm = colors.Normalize(fracs.min(), fracs.max())
    
    # for thisfrac, thispatch in zip(fracs, patches):
    #     color = plt.cm.viridis(norm(thisfrac))
    #     thispatch.set_facecolor(color)
    # ignore plotting all sting for now because Hamiltonian is huge
    # also no color distinction as the total sample is small
    axs[0].bar(list(dicts[0].keys()), list(dicts[0].values()), color='g', label='term grouped')
    axs[1].bar(list(dicts[1].keys()), list(dicts[1].values()), color='r', label='original qdrift')
    axs[0].legend()
    axs[1].legend()

# def sample_hist(sample,labels=['number', 'hopping'],c='green'):
#     sorted_list = sorted(sample)
#     sorted_counted = Counter(sorted_list)

#     range_length = list(range(max(sample))) # Get the largest value to get the range.
#     data_series = {}

#     for i in range_length:
#         data_series[i] = 0 # Initialize series so that we have a template and we just have to fill in the values.

#     for key, value in sorted_counted.items():
#         data_series[key] = value

#     data_series = pd.Series(data_series)
#     x_values = data_series.index

#     # you can customize the limits of the x-axis
#     # plt.xlim(0, max(some_list))
#     plt.bar(x_values, data_series.values, color=c,align='center')
#     plt.title('sample count for each group')
#     plt.ylabel('sample number')
#     plt.xticks(x_values, labels)
#     plt.xticks(rotation=70)
#     plt.legend()

#     plt.show() 