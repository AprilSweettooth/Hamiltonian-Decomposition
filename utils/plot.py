import numpy as np
import matplotlib.pyplot as plt

def cheat_plot(measurements):
    times = list(measurements.keys())
    exps = list(measurements.values())
    plt.plot(times, exps)

def evol_plot(times, exps, gate):
    plt.plot(times, exps)
