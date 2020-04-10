import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

import pyfid.simulation

np.random.seed(2)
outdir = os.path.join(os.path.dirname(__file__), "output")


sim = pyfid.simulation.rand_poly_frequency_exp_amplitude(
    f0=0.1,
    t1=143.0,
    deg=1,
    drift=0.1,
    duration=180.0,
    fs=10,
    snr=10
)

plt.figure()
plt.plot(sim.T, sim.simulate(), label="measurement")
plt.plot(sim.T, sim.amplitude(sim.T), label="amplitude")
plt.legend()
plt.xlabel("time (s)")

plt.twinx()
plt.plot(sim.T[:-1], np.diff(sim.phase(sim.T)), color="red")
plt.ylabel("frequency")

plt.tight_layout()
plt.savefig(os.path.join(outdir, "simulation.png"))
