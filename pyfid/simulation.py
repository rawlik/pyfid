import pickle
import os

import numpy as np

import pyfid.filtering as flter
from pyfid.cramer_rao import cramer_rao


class FIDsim:
    """
    Documentation of FIDsim class.
    """
    def __init__(self, amplitude_model, amplitude_parameters, phase_model,
        phase_parameters, sampling_rate, duration, snr, filter_func=None,
        filter_advance_time=None):
        """Documentation of the FID constructor.
        """

        self.sampling_rate = sampling_rate
        self.duration = duration
        self.filter_advance_time = filter_advance_time
        self.filter_func = filter_func
        # 1/s-t-n at the beginning
        # sine amplitude (NOT peak-to-peak) to standard deviation of noise
        self.noise = 1 / snr

        self.T = np.arange(0, duration, 1 / sampling_rate)

        self.phase_model = phase_model
        self.phase_parameters = phase_parameters
        self.amplitude_model = amplitude_model
        self.amplitude_parameters = amplitude_parameters

        def amplitude(t):
            return amplitude_model(t, *amplitude_parameters)
        self.amplitude = amplitude

        def phase(t):
            return phase_model(t, *self.phase_parameters)
        self.phase = phase


    def real_favg(self):
        """Calculate the real average frequency of the signal.
        """
        real_favg = (self.phase(self.T[-1]) - self.phase(self.T[0])) / (self.T[-1] - self.T[0]) / (2 * np.pi)

        return real_favg


    def simulate(self, n=1, squeeze=True):
        """Simulate an FID signal.

        Parameters
        n : int
            The number of signals to simulate. Default is 1.
        squeeze : int
            Whether to squeeze the dimension if n=1. If False always
            return a 2D array.

        Returns
        -------
        D : numpy array
            The array of measurements. The last axis are points.
            If n>1 or squeeze is True it is a 2D array and the first axis
            are different signals.
        """
        if self.filter_func is None:
            Tl = self.T
        else:
            Tl = np.r_[
                np.arange(self.T[0] - self.filter_advance_time, self.T[0], 1 / self.sampling_rate),
                self.T,
                np.arange(self.T[-1], self.T[-1] + self.filter_advance_time, 1 / self.sampling_rate)
                + 1 / self.sampling_rate]

        D = self.amplitude(Tl) * np.sin(self.phase(Tl))
        D = D + np.random.randn(n, Tl.size) * self.noise

        if self.filter_func is not None:
            D = self.filter_func(D, axis=-1)

        D = D[:, np.logical_and(Tl >= self.T[0], Tl <= self.T[-1])]

        if squeeze:
            D = np.squeeze(D)

        return D


    def cramer_rao_bound(self):
        """Calculate the Cramer-Rao bound: the lower limit
        of the precision with which the average frequency can be estimated.
        """
        def crmodel(t, parameters):
            amplitude_parameters = parameters[:len(self.amplitude_parameters)]
            phase_parameters = parameters[-len(self.phase_parameters):]

            return self.amplitude_model(t, *amplitude_parameters) * \
                np.sin(self.phase_model(t, phase_parameters))

        # evaluate the cramer_rao bound on the
        p0 = list(self.amplitude_parameters) + list(self.phase_parameters)
        CR = cramer_rao(crmodel, p0, self.T, self.noise)

        # we're not interested in additional model parameters:
        n_params = len(self.phase_parameters)
        CR = CR[:n_params, :n_params]

        I = np.arange(n_params)[::-1]
        Derivs = self.T[-1] ** I - self.T[0] ** I

        sfCR = np.sqrt(Derivs.dot(CR).dot(Derivs))
        sfCR *= 1 / (self.T[-1] - self.T[0]) / (2 * np.pi)

        return sfCR


def any_poly_frequency(frequency_coefficients, **kwargs):
    ph0 = np.random.rand()
    phase_parameters = np.poly1d(frequency_coefficients).integ(k=ph0).coeffs * 2 * np.pi

    def phase_model(t, *phase_parameters):
        return np.poly1d(phase_parameters)(t)

    return FIDsim(phase_model=phase_model, phase_parameters=phase_parameters, **kwargs)


def const_frequency_const_amplitude(f0, **kwargs):
    def amplitude_model(t):
        return 1

    amplitude_parameters = []

    return any_poly_frequency([f0], amplitude_model=amplitude_model,
        amplitude_parameters=amplitude_parameters, **kwargs)


def any_poly_frequency_exp_amplitude(t1, frequency_coefficients, **kwargs):
    def amplitude_model(t, t1):
        return np.exp(-t / t1)

    amplitude_parameters = [t1]

    return any_poly_frequency(frequency_coefficients,
        amplitude_model=amplitude_model,
        amplitude_parameters=amplitude_parameters, **kwargs)


def const_frequency_exp_amplitude(f0, t1, **kwargs):
    return any_poly_frequency_exp_amplitude(frequency_coefficients=[f0], **kwargs)


def rand_poly_frequency_coeffs(f0, deg, drift, duration, drift_linear_mean=0):
    # drift is inside one cycle
    if deg > 0:
        coeffs = 1 / np.sqrt(deg) * drift / duration ** np.r_[deg:0:-1]
        coeffs *= np.random.randn(deg)
        coeffs[-1] += drift_linear_mean
    else:
        coeffs = []
    coeffs = np.r_[coeffs, f0]

    return coeffs


def rand_poly_frequency_exp_amplitude(f0, t1, deg, drift, duration, **kwargs):
    coeffs = rand_poly_frequency_coeffs(f0, deg, drift, duration)

    return any_poly_frequency_exp_amplitude(t1, coeffs, duration=duration,
        **kwargs)


def any_poly_frequency_two_exp_amplitude(t1, t2, t1_to_t2_amplitudes_ratio,
        frequency_coefficients, **kwargs):

    def amplitude_model(t, t1, t2, r):
        A1 = r / (1. + r)
        A2 = 1. / (1. + r)
        return (A1 * np.exp(-t / t1) + A2 * np.exp(-t / t2))

    amplitude_parameters = [t1, t2, t1_to_t2_amplitudes_ratio]

    return any_poly_frequency(frequency_coefficients,
        amplitude_model=amplitude_model,
        amplitude_parameters=amplitude_parameters, **kwargs)


def const_frequency_two_exp_amplitude(f0, **kwargs):
    return any_poly_frequency_two_exp_amplitude(frequency_coefficients=[f0], **kwargs)


def rand_poly_frequency_two_exp_amplitude(f0, deg, drift, duration, **kwargs):
    coeffs = rand_poly_frequency_coeffs(f0, deg, drift, duration)

    return any_poly_frequency_two_exp_amplitude(coeffs, duration=duration, **kwargs)


def lin_frequency_two_exp_amplitude(f0, drift, **kwargs):
    coeffs = [drift / 180, f0]

    return any_poly_frequency_two_exp_amplitude(frequency_coefficients=coeffs, **kwargs)
