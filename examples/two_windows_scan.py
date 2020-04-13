import os

import numpy as np
import matplotlib.pyplot as plt

import pyfid.simulation
import pyfid.estimation
import pyfid.optimization
import pyfid.nEDMatPSI


np.random.seed(0)
outdir = os.path.join(os.path.dirname(__file__), "output")

sim_gen = lambda: pyfid.simulation.rand_poly_frequency_two_exp_amplitude(
    f0=7.8,
    t1=14.3,
    t2=1.2,
    t1_to_t2_amplitudes_ratio=0.1,
    deg=1,
    drift=0.1,
    duration=18.0,
    fs=100,
    snr=144)

win_lengths = np.linspace(2, 12, num=10)
results = []

for win_length in win_lengths:
    estimator = lambda T, D, sD: pyfid.estimation.two_windows(
        T=T, D=D, sD=sD,
        submethod='phase',
        prenormalize=False,
        double_exp=(True, False),
        phase_at_end=True,
        win_len=(win_length / 20, win_length),
        verbose=False)

    accuracy, precision, saccuracy, sprecision = \
        pyfid.optimization.accuracy_and_precision_different_sims(
            sim_gen=sim_gen,
            estimator=estimator,
            nsimulations=5,
            nsignals=5,
            full_output=True)

    results.append((accuracy, precision, saccuracy, sprecision))

results = np.array(results)

fig = plt.figure()
plt.errorbar(win_lengths, np.abs(results[:, 0]),
    yerr=results[:, 2], fmt=".", label="accuracy")
plt.errorbar(win_lengths, results[:, 1],
    yerr=results[:, 3], fmt=".", label="precision")

plt.legend()

plt.yscale('log', nonposy='clip')
plt.xlabel('Second window size (s)')
plt.ylabel('Hz')

plt.tight_layout()
plt.savefig(os.path.join(outdir, "two_windows_scan.png"))
