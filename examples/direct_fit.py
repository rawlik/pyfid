"""Demonstrate the direct fit method.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

import pyfid.simulation
import pyfid.estimation
import pyfid.nEDMatPSI

np.random.seed(0)
outdir = os.path.join(os.path.dirname(__file__), "output")

sim = pyfid.simulation.rand_poly_frequency_two_exp_amplitude(
    fs=pyfid.nEDMatPSI.fs,
    f0=pyfid.nEDMatPSI.filter_f0,
    duration=pyfid.nEDMatPSI.duration,
    drift=pyfid.nEDMatPSI.drift,
    deg=0,
    snr=144,
    t1=pyfid.nEDMatPSI.t1,
    t2=pyfid.nEDMatPSI.t2,
    t1_to_t2_amplitudes_ratio=pyfid.nEDMatPSI.t1_to_t2_amplitudes_ratio,
    filter_advance_time=1,
    filter_func=pyfid.nEDMatPSI.nEDMfilter)

D = sim.simulate(random_phase=True)

fig = plt.figure()

def do_plot(ax):
    ax.errorbar(sim.T, D, yerr=sim.sigma(), fmt='k,', capsize=0)
    f, sf, details = pyfid.estimation.direct_fit(
        sim.T, D, sim.sigma(),
        model_key="double_damped_sine_DC",
        plot_ax=ax)

do_plot(fig.add_subplot(211))
do_plot(fig.add_subplot(223))
plt.xlim(0, 1)
do_plot(fig.add_subplot(224))
plt.xlim(179, 180)
plt.ylim(-0.05, 0.05)

plt.tight_layout()
plt.savefig(os.path.join(outdir, "direct_fit.png"))
