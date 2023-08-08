import matplotlib.pyplot as plt
from scipy import linalg
import numpy as np

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

def compare_spectral_error(sample_number, U, Us, Us_orig):
    plt.plot(sample_number,[np.abs(linalg.eig(u - u_exc)[0]).max() for u,u_exc in zip(Us,U)], label='term grouping')
    plt.plot(sample_number,[np.abs(linalg.eig(u - u_exc)[0]).max() for u,u_exc in zip(Us_orig,U)], label='original drift')
    plt.title('spectral error over sample number')
    plt.xlabel('Sample number')
    plt.ylabel('spectral error') 
    plt.legend()