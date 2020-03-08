import os

import numpy as np
import matplotlib.pyplot as plt

import pyfid.cramer_rao


outdir = os.path.join(os.path.dirname(__file__), "output")

def sig(t, a, f, ph):
    return a * np.sin(2 * np.pi * f * t + ph)

p0 = [2.12, 8, 0]
noise = 0.09

T = np.arange(-1, 1, 0.01)
C = pyfid.cramer_rao.cramer_rao(sig, p0, T, noise,
    show_plot=True, quad_precision=True)
plt.savefig(os.path.join(outdir, "cramer_rao.png"))

print('Inverse of the Fisher information matrix:')
print(C)
print('numerical: ', C.diagonal())

Cmc = pyfid.cramer_rao.cramer_rao_monte_carlo(sig, p0, T, noise,
    show_plot=True)
print('monte-carlo: ', C.diagonal())

var_a = noise**2 / T.size * 2
var_w = 12 * noise**2 / ( p0[0]**2 * (T[1]-T[0])**2 * \
    (T.size**3 - T.size) ) * 2
var_f = var_w / (2*np.pi)**2
var_ph = noise**2 / p0[0]**2 / T.size * 2

print('teoretical:', np.array([var_a, var_f, var_ph]))
# ref: D. Rife, R. Boorstyn "Single-Tone Parameter Estimation from
# Discrete-Time Observations", IEEE Transactions on Information Theory
# 20, 5, 1974

plt.savefig(os.path.join(outdir, "cramer_rao_signal.png"))

