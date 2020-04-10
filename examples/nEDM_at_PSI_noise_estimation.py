import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.optimize

import pyfid.filtering as flter
import pyfid.nEDMatPSI


np.random.seed(0)
outdir = os.path.join(os.path.dirname(__file__), "output")


def filter_frequency_response(freqs):
    _w, h = scipy.signal.freqz(pyfid.nEDMatPSI.filter_b, pyfid.nEDMatPSI.filter_a,
        worN=freqs * np.pi / 0.5 / pyfid.nEDMatPSI.fs)

    return abs(h) ** 2


def noise_from_signal(T, D, fs, plot=False):
    """Ref. Sec. 5.1 of Martin Fertl PhD Thesis (2013)."""
    FFTfreqs = np.fft.fftfreq(T.size, 1 / fs)
    freqs = FFTfreqs[FFTfreqs >= 0]

    def normedfft(X):
        return abs(np.fft.fft(X))[FFTfreqs >= 0] * 2 / T.size

    signalFFT = normedfft(D)
    windowedFFT = normedfft(D * scipy.signal.windows.hann(T.size) / 0.5 * np.sqrt(2/3))
    filterFFT = filter_frequency_response(freqs)

    mask = (freqs > 7.5) & (freqs < 8.1)
    mask = mask | (freqs > 12) & (freqs < 13)
    mask = mask | (freqs > 14) & (freqs < 17)
    mask = mask | (freqs < 2) | (freqs > 30)
    mask = ~mask

    def chi2(x):
        A, C = x
        return sum((windowedFFT[mask] - (A * filterFFT[mask] + C))**2)

    A, C = scipy.optimize.minimize(chi2, [1.,1.]).x

    noise_amplitude = np.amax(A * filterFFT + C) * fs

    if plot:
        plt.figure("Noise level from the signal", figsize=(10, 5.5))

        plt.plot(freqs, signalFFT, 'r.-', lw=1.5, ms=4,
             label='filtered signal power spectrum')
        plt.plot(freqs, windowedFFT, 'k-', lw=1, ms=4,
             label='windowed signal power spectrum')
        plt.plot(freqs[mask], windowedFFT[mask], '.', lw=1.5, ms=4, color="orange",
             label='windowed signal power spectrum')
        plt.plot(freqs, A * filterFFT + C, '-', lw=1.5, color="lightgreen",
             label="filter's frequency response")

        plt.xlabel('frequency (Hz)')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(3, 20)
        # yticks([])
        plt.grid()
        plt.legend(loc='upper right', fontsize='medium')

        plt.tight_layout()

    return noise_amplitude


T = np.arange(0, 180, 1 / pyfid.nEDMatPSI.fs)
FFTfreqs = np.fft.fftfreq(T.size, 1 / pyfid.nEDMatPSI.fs)
freqs = FFTfreqs[FFTfreqs >= 0]

true_noise_amplitude = 10
N = np.random.randn(T.size) * true_noise_amplitude
S = 500 * np.exp(-T / 60) * np.sin(2 * np.pi * 7.85 * T)

calculated_noise_amplitude = noise_from_signal(T, pyfid.nEDMatPSI.nEDMfilter(S + N), pyfid.nEDMatPSI.fs, plot=True)

print("True noise amplitude: ", true_noise_amplitude)
print("Calculated noise amp. ", calculated_noise_amplitude)

plt.tight_layout()
plt.savefig(os.path.join(outdir, "nEDM_at_PSI_noise_estimation.png"))
