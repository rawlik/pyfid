import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.optimize

import pyfid.filtering

# seed the random number generator
np.random.seed(0)
outdir = os.path.join(os.path.dirname(__file__), "output")


# sampling frequency
fs = 100.

# the parameters of the filter
f0 = 7.852
lowcut = 7.125
highcut = 8.68
order = 1

# define the filtering function
nyq = 0.5 * fs
low = lowcut / nyq
high = highcut / nyq
b, a = scipy.signal.butter(order, [low, high], btype='band')


def nEDMfilter(Y):
    return scipy.signal.filtfilt(b, a, Y)


# load the measurement data
# from the thesis of Perkowski
D = np.loadtxt(os.path.join(os.path.dirname(__file__), "8Hz.txt"))

# sort by first column (frequency) and transpose
D = D[D[:,0].argsort()].T
Fm, Ym, sYm = D[0], D[1], D[2]

# average results for same frequency
nYm, snYm = [], []
for fm in np.unique(Fm):
    I = Fm == fm
    nYm.append(np.average(Ym[I], weights=1/sYm[I]**2))
    snYm.append(np.sqrt(sum(sYm[I])))

nFm = np.unique(Fm)
nYm = np.array(nYm)
snYm = np.array(snYm)

# interpolate result with spline to find proper norm
spline = scipy.interpolate.UnivariateSpline(nFm, nYm, w=1/snYm**2)
res = scipy.optimize.minimize_scalar(lambda f: -spline(f),
    method='Bounded', bounds=(Fm[0], Fm[-1]))
spline_norm = -1 / res.fun

Fm = D[0]
Ym = spline_norm * D[1]
sYm = spline_norm * D[2]


# Plot the frequency response
plt.figure(figsize=(8,5))
w, h = scipy.signal.freqz(b, a, worN=2000)
plt.plot((fs * 0.5 / np.pi) * w, abs(h)**2, 'k-', lw=2, label='power spectrum')
plt.plot((fs * 0.5 / np.pi) * w, abs(h), 'k--', lw=2, label='magnitude')
plt.axvline(f0, color='black', ls='--',label='central frequency')
# plot measured filter frequency response
plt.errorbar(Fm, Ym, yerr=sYm, fmt='x', ms=8, capsize=0, color='black',
    label='measured frequency response')
plt.xlabel('Frequency (Hz)')
plt.legend(frameon=False)
plt.xlim(xmax=25)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "filter_frequency_response.png"))


# Plot filtered noise
plt.figure('filtered white noise', figsize=(10,6))
plt.subplot(211)
# Filter noise.
T = np.arange(0, 15, 1/fs)
N = np.random.randn(T.size)
S = N
F = nEDMfilter(S)
C = T<5
plt.plot(T[C], S[C], 'k.-', ms=4, alpha=0.5, label='white noise')
plt.plot(T[C], F[C], 'r.-', lw=1.5, ms=4, label='filtered white noise')
plt.xlabel('time (s)')
plt.yticks([])
plt.legend(loc='upper right')

plt.subplot(212)
FFT = abs(np.fft.fft(N))
FFTfreqs = np.fft.fftfreq(T.size, 1/fs)
FFT = FFT[FFTfreqs >= 0]
FFTfreqs = FFTfreqs[FFTfreqs >= 0]
plt.plot(FFTfreqs, FFT, 'k.-', ms=4, alpha=0.5, label='white noise FT')

FFT = abs(np.fft.fft(F))
FFTfreqs = np.fft.fftfreq(T.size, 1/fs)
FFT = FFT[FFTfreqs >= 0]
FFTfreqs = FFTfreqs[FFTfreqs >= 0]
plt.plot(FFTfreqs, FFT, 'r.-', lw=1.5, ms=4, label='filtered white noise FT')

F = (fs * 0.5 / np.pi) * w
plt.plot(F, abs(h)**2 * max(FFT), 'k--', lw=2, label="filter's frequency response")

plt.xlabel('frequency (Hz)')
plt.yticks([])
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(outdir, "filter_noise_filtering.png"))


# plot a filtered signal
sn = 5
T = np.arange(0, 10, 1/fs)
N = np.random.randn(T.size) / sn
# TODO use a simulation package here
T1 = T[T>1]
Sin = np.sin(2*np.pi*8 * T1 + np.random.rand()*2*np.pi ) * np.exp(-T1/2.)
S = np.r_[np.zeros(T.size - T1.size), Sin]
Y = S + N
Z = nEDMfilter(Y)
plt.figure(figsize=(20,10))
plt.subplot(211)
plt.plot(T, S, 'k.-', ms=4, label='signal without noise')
plt.plot(T, Y, 'b.-', ms=4, label='signal + noise')
plt.legend(loc='best')
plt.subplot(212)
plt.plot(T, S, 'k.-', ms=4, label='signal without noise')
plt.plot(T, Z, 'r.-', ms=4, label='filtered signal')
plt.title('')
plt.legend(loc='best')
plt.savefig(os.path.join(outdir, "filter_signal_filtering.png"))


# visualise the filter
plt.figure()
F = (fs * 0.5 / np.pi) * w

Noise = 0*F + 1e-3
Signal = scipy.stats.norm.pdf(F, f0, 0.1) / max(scipy.stats.norm.pdf(F, f0, 0.1))

plt.plot(F, abs(h)**2, 'k-.', lw=1, label="filter's frequency responce")

plt.fill_between(F, Noise, color='red', alpha=0.3)
plt.fill_between(F, Signal + Noise, Noise, color='green', alpha=0.5)
plt.plot(F, Signal + Noise, 'b-', lw=3, label='measured signal')
plt.plot(F, (Signal + Noise) * abs(h)**2, 'k--', lw=2, label='filtered signal')
plt.fill_between(F, np.minimum((Signal + Noise) * abs(h)**2, Noise),
    facecolor='None', hatch='///', edgecolor='red', lw=0)

plt.annotate('actual signal', xy=(f0+0.2, 1e-2),
    xytext=(100, -10),
    horizontalalignment='center',
    verticalalignment='center',
    textcoords='offset points',
    arrowprops={'arrowstyle':'->', 'lw':2},
    fontsize=18)

plt.annotate('noise', xy=(17, 1e-4),
    horizontalalignment='center',
    verticalalignment='center',
    fontsize=25)
plt.annotate('noise', xy=(2.5, 1e-4),
    horizontalalignment='center',
    verticalalignment='center',
    fontsize=25)

plt.annotate('noise in\nfiltered signal', xy=(f0+0.3, 4e-5),
    horizontalalignment='center',
    verticalalignment='center',
    fontsize=20)

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (a.u.)')
plt.yscale('log', nonposy='clip')
plt.ylim(ymin=1e-5)
plt.xlim(xmin=0, xmax=20)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(outdir, "filter_visualisation.png"))

# plt.show()
