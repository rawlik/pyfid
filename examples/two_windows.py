import os

import matplotlib.pyplot as plt
import numpy as np

import pyfid.simulation
import pyfid.estimation
import pyfid.nEDMatPSI


np.random.seed(0)
outdir = os.path.join(os.path.dirname(__file__), "output")

sim = pyfid.simulation.lin_frequency_two_exp_amplitude(
    fs=pyfid.nEDMatPSI.fs,
    f0=pyfid.nEDMatPSI.filter_f0,
    duration=pyfid.nEDMatPSI.duration,
    drift=pyfid.nEDMatPSI.drift,
    snr=144,
    t1=pyfid.nEDMatPSI.t1,
    t2=pyfid.nEDMatPSI.t2,
    t1_to_t2_amplitudes_ratio=pyfid.nEDMatPSI.t1_to_t2_amplitudes_ratio,
    filter_advance_time=1,
    filter_func=pyfid.nEDMatPSI.nEDMfilter)

D = sim.simulate()

fig = plt.figure()

f, sf, details = pyfid.estimation.two_windows(
    T=sim.T,
    D=D,
    sD=sim.sigma(),
    submethod='phase',
    prenormalize=False,
    double_exp=(True, False),
    phase_at_end=True,
    win_len=(1, 3),
    plot_fig=fig,
    verbose=False)

plt.tight_layout()
plt.savefig(os.path.join(outdir, "two_windows.png"))
