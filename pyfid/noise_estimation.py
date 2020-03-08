import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz, hann
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import norm

import pyfid.filtering as flter


def filter_frequency_response(freqs):
    b, a = flter.butter_bandpass(flter.lowcut, flter.highcut, flter.fs,
                                 order=flter.order)
    w, h = freqz(b, a, worN=freqs * np.pi / 0.5 / flter.fs)

    return abs(h) ** 2


def noise_from_signal(T, D, fs, plot=False):
    """Ref. Sec. 5.1 of Martin Fertl PhD Thesis (2013)."""
    FFTfreqs = np.fft.fftfreq(T.size, 1 / fs)
    freqs = FFTfreqs[FFTfreqs >= 0]

    def normedfft(X):
        return abs(np.fft.fft(X))[FFTfreqs >= 0] * 2 / T.size

    signalFFT = normedfft(D)
    windowedFFT = normedfft(D * hann(T.size) / 0.5 * np.sqrt(2/3))
    filterFFT = filter_frequency_response(freqs)

    mask = (freqs > 7.5) & (freqs < 8.1)
    mask = mask | (freqs > 12) & (freqs < 13)
    mask = mask | (freqs > 14) & (freqs < 17)
    mask = mask | (freqs < 2) | (freqs > 30)
    mask = ~mask

    def chi2(x):
        A, C = x
        return sum((windowedFFT[mask] - (A * filterFFT[mask] + C))**2)

    A, C = minimize(chi2, [1.,1.]).x

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


def plot_filtered_white_noise():
    figure('filtered white noise', figsize=(10, 5.5))

    # subplot(211)
    # Filter noise.
    flter.fs = 100
    T = np.arange(0, 180, 1 / flter.fs)
    FFTfreqs = np.fft.fftfreq(T.size, 1 / flter.fs)
    freqs = FFTfreqs[FFTfreqs >= 0]

    N = randn(T.size)
    S = 100 * exp(-T / 60) * sin(2 * np.pi * 7.85 * T)


if __name__ == "__main__":
    T = np.arange(0, 180, 1 / flter.fs)
    FFTfreqs = np.fft.fftfreq(T.size, 1 / flter.fs)
    freqs = FFTfreqs[FFTfreqs >= 0]

    true_noise_amplitude = 10
    N = np.random.randn(T.size) * true_noise_amplitude
    S = 500 * np.exp(-T / 60) * np.sin(2 * np.pi * 7.85 * T)

    calculated_noise_amplitude = noise_from_signal(T, flter.flter(S + N), flter.fs, plot=True)

    print("True noise amplitude: ", true_noise_amplitude)
    print("Calculated noise amp. ", calculated_noise_amplitude)

    plt.show()
