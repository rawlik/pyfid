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

