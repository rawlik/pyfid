"""
This module holds the parameters specific to the FID of the mercury
comagnetometer's signal in the nEDM-at-PSI experiment.
"""
import numpy as np
import scipy.signal

import pyfid.simulation
import pyfid.estimation
import pyfid.optimization


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
    return scipy.signal.lfilter(filter_b, filter_a, Y)


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


def optimize_window_size(noise, drift_mean, drift_std, t1, t2,
        t1_to_t2_amplitudes_ratio, cycle_duration, first_window_divisor=20,
        f_change_degree=1, Ndrifts=10, Nfor_drift=10, size_precision=1,
        Npoints=10, with_filter=True, data_amplitude=None,
        plot_afterwards=False, verbose=False):
    """
    This function is very specific to the nEDM experiment.
    """
    log_scan = True

    if data_amplitude is None:
        sim_gen = lambda: pyfid.simulation.rand_poly_frequency_two_exp_amplitude(
            f0=pyfid.nEDMatPSI.filter_f0,
            t1=t1,
            t2=t2,
            t1_to_t2_amplitudes_ratio=t1_to_t2_amplitudes_ratio,
            deg=f_change_degree,
            drift=drift_std * cycle_duration,
            drift_linear_mean=drift_mean,
            duration=cycle_duration,
            fs=pyfid.nEDMatPSI.fs,
            snr=1 / noise,
            filter_func=pyfid.nEDMatPSI.nEDMfilter if with_filter else None)
    else:
        # TODO
        raise NotImplementedError("data amplitude not implemented yet")


    # first test, whether the direct fit is okay
    direct_fit_estimator = lambda T, D, sD: pyfid.estimation.direct_fit(
        T, D, sD, double_exp=True)

    accuracy, precision, saccuracy, sprecision = \
        pyfid.optimization.accuracy_and_precision_different_sims(
            sim_gen=sim_gen,
            estimator=direct_fit_estimator,
            nsimulations=Ndrifts,
            nsignals=Nfor_drift,
            full_output=True)

    conservative_precision = precision + 3 * sprecision
    conservative_accuracy = np.abs(accuracy) - 3 * saccuracy

    direct_fit_ok = conservative_accuracy < conservative_precision

    if direct_fit_ok:
        return 0
    else:
        estimator = lambda p, T, D, sD: pyfid.estimation.two_windows(
            T=T, D=D, sD=sD,
            submethod='phase',
            prenormalize=False,
            double_exp=(True, False),
            phase_at_end=True,
            win_len=(p / first_window_divisor, p),
            verbose=False)

        max_second_size = cycle_duration * first_window_divisor / (first_window_divisor + 1)

        bisect_to_max = lambda max_win_size: pyfid.optimization.bisect_parameter(
            sim_gen=sim_gen,
            estimator=estimator,
            p_min=3,
            p_max=max_win_size,
            p_tol=size_precision,
            nsimulations=Ndrifts,
            nsignals=Nfor_drift)

        try:
            optimum = bisect_to_max(max_second_size)
        except ValueError:
            try:
                optimum = bisect_to_max(0.8 * max_second_size)
            except ValueError:
                try:
                    optimum = bisect_to_max(0.6 * max_second_size)
                except ValueError:
                    try:
                        optimum = bisect_to_max(0.4 * max_second_size)
                    except ValueError:
                        # the final fallback value
                        optimum = 10.

            return optimum
