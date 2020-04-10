import numpy as np
import scipy.signal


def noise_suppression_factor(filter_function, fs, N=1000):
    """
    Estimate the suppression factor of the filter based on filtering
    a sample of N normally distributed numbers.

    Parameters
    ----------
    filter_function : callable
        A function that will be used to filter the array.
    fs : float
        The sampling frequency

    Returns
    -------
    supp : float
        The ratio of the standard deviation of white noise to filtered white
        noise.
    """
    T = np.arange(0, N, 1/fs)
    N = np.random.randn(T.size)
    s = np.std(N)
    F = filter_function(N)
    supp = s / np.std(F)

    return supp


def filter_frequency_response(freqs, b, a, fs):
    """
    The frequency response of a filter.

    Parameters
    ----------
    freqs : array
        The frequencies to evaluate the response for.
    b, a : arrays
        The filter taps.
    fs : float
        The sampling frequency in Hz.

    Returns
    -------
    array
        The frequency response.
    """
    _w, h = scipy.signal.freqz(b, a,
        worN=freqs * np.pi / 0.5 / fs)

    return abs(h) ** 2


def noise_from_signal(D, b, a, fs, mask, full_output=False):
    """
    Estimate the amplitude of noise in an FID.

    It is asummed that the signal is white noise that has been filtered.
    The filter's frequency response is fitted to the spectrum of the signal
    with selected frequencies masked.

    Parameters
    ----------
    D : array
        The signal to be analysed
    b, a : arrays
        The taps of the filter.
    mask : callable
        A function to mask the relevant not-noise frequencies in the signal.
    full_output : bool
        Whether to return all variables. By default only `noise_amplitude`
        is returned.
        sm, A * filterFFT + C

    Returns
    -------
    noise_amplitude : float
        The amplitude of noise.
    freqs : array
        The frequencies considered.
    signalFFT : array
        The FFT of the signal.
    windowedFFT : array
        The signal FFT windowed with a hann window.
    m : bool array
        The mask applied to the `freqs` array. Returns True for frequencies
        to be masked.
    fitted_frequency_response : array
        The fitted filter frequency response.

    References
    ----------
    .. [1] Sec. 5.1 of Martin Fertl PhD Thesis (2013).
    """
    FFTfreqs = np.fft.fftfreq(D.size, 1 / fs)
    freqs = FFTfreqs[FFTfreqs >= 0]

    def normedfft(X):
        return abs(np.fft.fft(X))[FFTfreqs >= 0] * 2 / D.size

    signalFFT = normedfft(D)
    windowedFFT = normedfft(D * scipy.signal.windows.hann(D.size) / 0.5 * np.sqrt(2/3))
    filterFFT = filter_frequency_response(freqs, b, a, fs)

    m = np.array([ ~mask(f) for f in freqs ])

    def chi2(x):
        A, C = x
        return sum((windowedFFT[m] - (A * filterFFT[m] + C))**2)

    A, C = scipy.optimize.minimize(chi2, [1.,1.]).x
    fitted_frequency_response = A * filterFFT + C

    noise_amplitude = np.amax(fitted_frequency_response) * fs

    if full_output:
        return noise_amplitude, freqs, signalFFT, windowedFFT, m, fitted_frequency_response
    else:
        return noise_amplitude
