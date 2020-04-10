import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.optimize

import pyfid.filtering
import pyfid.nEDMatPSI


np.random.seed(0)
outdir = os.path.join(os.path.dirname(__file__), "output")


T = np.arange(0, 180, 1 / pyfid.nEDMatPSI.fs)
true_noise_amplitude = 10
N = np.random.randn(T.size) * true_noise_amplitude
S = 500 * np.exp(-T / 60) * np.sin(2 * np.pi * 7.85 * T)

def mask(f):
    mask = (f > 7.5) & (f < 8.1)
    mask = mask | (f > 12) & (f < 13)
    mask = mask | (f > 14) & (f < 17)
    mask = mask | (f < 2) | (f > 30)

    return mask

calculated_noise_amplitude, freqs, signalFFT, windowedFFT, m, \
    fitted_frequency_response = pyfid.filtering.noise_from_signal(
        D=pyfid.nEDMatPSI.nEDMfilter(S + N),
        b=pyfid.nEDMatPSI.filter_b,
        a=pyfid.nEDMatPSI.filter_a,
        fs=pyfid.nEDMatPSI.fs,
        mask=mask,
        full_output=True)

plt.figure("Noise level from the signal", figsize=(10, 5.5))

plt.plot(freqs, signalFFT, 'r.-', lw=1.5, ms=4,
        label='filtered signal power spectrum')
plt.plot(freqs, windowedFFT, 'k-', lw=1, ms=4,
        label='windowed signal power spectrum')
plt.plot(freqs[m], windowedFFT[m], '.', lw=1.5, ms=4, color="orange",
        label='windowed signal power spectrum')
plt.plot(freqs, fitted_frequency_response, '-', lw=1.5, color="lightgreen",
        label=f"filter's frequency response, max={calculated_noise_amplitude:.2f}")

plt.xlabel('frequency (Hz)')
plt.xscale('log')
plt.yscale('log')
plt.xlim(3, 20)
plt.ylim(1e-6, 1e3)
plt.grid()
plt.legend(loc='upper right', fontsize='medium')

plt.tight_layout()

print("True noise amplitude: ", true_noise_amplitude)
print("Calculated noise amp. ", calculated_noise_amplitude)

plt.tight_layout()
plt.savefig(os.path.join(outdir, "nEDM_at_PSI_noise_estimation.png"))
