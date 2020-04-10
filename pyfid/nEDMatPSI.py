"""
This module holds the parameters specific to the FID of the mercury
comagnetometer's signal in the nEDM-at-PSI experiment.
"""

import scipy.signal


# sampling frequency
fs = 100.
"The sampling frequency."

# parameters of the signal
duration = 180.
"The duration of the FID."

drift = 9.6e-6
"The magnitude of frequency drift over the whole FID, in Hz."

snr = 144
"The signal-to-noise ratio at the start of the FID."

t1 = 143.
"The long decay constant, in seconds"

t2 = 12.
"The short decay constant, in seconds"

t1_to_t2_amplitudes_ratio = 0.1
"The ratio of amplitudes associated with `t1` and `t2`."

# the parameters of the filter
filter_f0 = 7.852
"The central frequency of the filter."

filter_lowcut = 7.125
"The lowcut frequency of the filter."

filter_highcut = 8.68
"The highcut frequency of the filter."

filter_order = 1
"The order of the filter."

# define the filtering function
_nyq = 0.5 * fs
_low = filter_lowcut / _nyq
_high = filter_highcut / _nyq
filter_b, filter_a = scipy.signal.butter(filter_order, [_low, _high],
    btype="band")
"The taps of the filter."

def nEDMfilter(Y):
    """
    Filter the array as in the nEDM experiment.

    Parameters
    ----------
    Y : array

    Returns
    -------
    np.ndarray
        Filtered array
    """
    return scipy.signal.filtfilt(filter_b, filter_a, Y)


def signal_mask(f):
    """
    Mask the frequencies where the signal is.

    Used is estimating the amplitude of noise based on a signal.

    Parameters
    ----------
    f : float or array
        frequency

    Returns
    -------
    mask : bool or array
    """
    mask = (f > 7.5) & (f < 8.1)
    mask = mask | (f > 12) & (f < 13)
    mask = mask | (f > 14) & (f < 17)
    mask = mask | (f < 2) | (f > 30)

    return mask
