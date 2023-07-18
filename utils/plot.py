import matplotlib.pyplot as plt

def cheat_plot(measurements, labels):
    times = list(measurements.keys())
    exps = list(measurements.values())
    plt.plot(times, exps, label=labels)
    plt.legend()
    plt.show()

def evol_plot(times, exps, gate, infidelity, labels):
    fig, axs = plt.subplots(1, 3, sharex='col', figsize=(3*6, 4))
    axs[0].set_title('Projetion on initial state')
    axs[0].set_ylim([-0.05,1.05])
    axs[0].plot(times, exps, label=labels[0])
    axs[1].set_title('CX gate count')
    axs[1].plot(times, gate, label=labels[1])
    axs[2].set_title('infidelity evolution')
    axs[2].plot(times, infidelity, label=labels[2])
    for i in range(3):
        axs[i].legend() 
    plt.show()

def compare_plot(times, exps, gates, infidelities, labels):
    fig, axs = plt.subplots(1, 3, sharex='col', figsize=(3*6, 4))
    for i in range(len(exps)):
        axs[0].plot(times, exps[i], label=labels[i])
        axs[1].plot(times, gates[i], label=labels[i])
        axs[2].plot(times, infidelities[i], label=labels[i])
    axs[0].set_xlabel(r'Time (1/J)')
    axs[0].set_ylabel(r'Prob')
    axs[0].set_ylim([-0.05,1.05])
    axs[1].set_ylabel(r'# cx-gates')
    axs[2].set_ylabel(r'# infidelity')
    for i in range(3):
        axs[i].legend()
    plt.show()

def exp_plot(measurements, labels):
    times = list(measurements.keys())
    exps = list(measurements.values())
    plt.plot(times, exps, label=labels)
    plt.legend()
    plt.show()