"""
The module `estimation` holds various methods of estimating the frequency
of an FID signal.
"""
import inspect

import numpy as np
import scipy.optimize
import scipy.stats
import warnings

# abs is to ensure that sine is not mulitlied by -1 which artificialy shifts
# phase by pi
models = {
    "sine_unit_amplitude":
        lambda t, f, ph: np.sin(2 * np.pi * f * t + ph),
    "sine_model":
        lambda t, f, ph, a: abs(a) * np.sin(2 * np.pi * f * t + ph),
    "sine_model_DC":
        lambda t, f, ph, a, DC: abs(a) * np.sin(2 * np.pi * f * t + ph) + DC,
    "line_sine":
        lambda t, f, ph, aa, ab: (aa * t + abs(ab)) * np.sin(2 * np.pi * f * t + ph),
    "line_sine_DC":
        lambda t, f, ph, aa, ab, DC: (
            aa * t + abs(ab)) * np.sin(2 * np.pi * f * t + ph) + DC,
    "damped_sine":
        lambda t, f, ph, aa, ab: (
            abs(aa) * np.exp(ab * t)) * np.sin(2 * np.pi * f * t + ph),
    "double_damped_sine":
        lambda t, f, ph, aa, ab, ba, bb: (
            abs(aa) * np.exp(ab * t) + abs(ba) * np.exp(bb * t)) * np.sin(2 * np.pi * f * t + ph),
    "double_damped_sine_DC":
        lambda t, f, ph, aa, ab, ba, bb, DC: (
            abs(aa) * np.exp(ab * t) + abs(ba) * np.exp(bb * t)) * np.sin(2 * np.pi * f * t + ph) + DC,
    "damped_sine_DC":
        lambda t, f, ph, aa, ab, DC: (
            abs(aa) * np.exp(ab * t)) * np.sin(2 * np.pi * f * t + ph) + DC,
}

class EstimationDetails:
    """A class to hold detailed results of the frequency estimation.
    The content may be different for every method.
    """

    def get_string_description(self):
        s = ''
        for attribute in dir(self):
            if attribute[:2] != '__':
                value = getattr(self, attribute)
                if 'get_string_description' in dir(value):
                    s += '%s:\n' % attribute
                    description = value.get_string_description()
                    s += '    ' + description[:-1].replace('\n', '\n    ')
                    s += '\n'
                elif not callable(value):
                    s += '%s: %s\n' % (attribute, value)
        return s

    def get_dictionary(self):
        d = {}
        for attribute in dir(self):
            if attribute[:2] != '__':
                value = getattr(self, attribute)
                if 'get_dictionary' in dir(value):
                    d[attribute] = value.get_dictionary()
                elif not callable(value):
                    d[attribute] = value
        return d


def fit_sine(X, Y, model_key, sigma=None, plot_ax=None, optimize_var_ph=False):
    """
    Fit a an oscillating-signal model to data.

    Uses of the the models defined in `pyfid.estimation.models`, for which
    a robust initial parameter guessing is implemented. The final fit is
    performed by the `scipy.optimize.curve_fit` function.

    Parameters
    ----------
    X : array
        The array of x values.
    Y : array
        The array of y values
    model_key : str
        Has to refer to one of the built-in models define in the
        dictionary `pyfid.estimation.models`.
    sigma : array, optional
        The array of uncertainties for `Y` values used to determine weights
        in fitting. Passed on to `scipy.optimize.curve_fit`. Default is
        an array of ones.
    plot_ax : matplotlib.Axes, optional
        If passed the result of the fit will be plotted there.
    optimize_var_ph : bool, optional
        If True it will shift the array `X` such, that the point corresponding
        to `x=0` is where the estimators of phase and frequency are
        uncorrelated. Care has to be taken when interpreting other
        time-dependent model parameters! Default is False.

    Retruns
    -------
    popt : array
        The array of the optimal parameters.
    pcov : array
        The estimator of the covariance matrix of the parameters in the
        minimum.
    details : EstimationDetails
        An object holding auxillary information about the fit.
    """
    # check if size of data is larger than number of model parameters
    nparms = len(inspect.signature(models[model_key]).parameters) - 1
    if len(X) <= nparms:
        # don't even attempt a fit
        details = EstimationDetails()
        details.model_name = model_key
        details.model_function = models[model_key]
        details.chi2 = np.nan
        details.deg_of_freedom = np.nan
        details.reduced_chi2 = np.nan
        details.CL = np.nan
        details.fit_ok = False
        details.popt = np.nan
        details.pcov = np.nan
        details.t0 = np.nan
        return np.ones(nparms) * np.nan, np.ones([nparms, nparms]) * np.nan, details

    fit_ok = True

    model = models[model_key]

    sigma_provided = sigma is not None
    if not sigma_provided:
        sigma = 0.0 * X + 1

    # FFT for frequency estimate
    FFT = np.absolute(np.fft.fft(Y))
    FFT = FFT[: len(FFT) // 2]
    FFT[0] = 0.0

    freqs = np.fft.fftfreq(Y.size, X[1] - X[0])[: len(FFT)]
    f0 = abs(freqs[np.argmax(FFT)])

    if f0 == 0:
        f0 = 8

    # if plot_ax is not None:
    # plot FFT
    #     subplot(211)
    #     plot(freqs, FFT, 'k', label='data FFT')
    #     axvline(f0, lw=3, label='FFT max')
    #     grid()
    #     legend()

    # first guess phase
    if model_key not in ['sine_unit_amplitude']:
        ph_model = lambda x, ph: models['sine_model'](
            x, f0, ph, (max(Y) - min(Y)) / 2)
    else:
        ph_model = lambda x, ph: models['sine_unit_amplitude'](x, f0, ph)

    i = int(2 / f0 / (X[1] - X[0]))
    init_ph, sph = scipy.optimize.curve_fit(
        ph_model, X[:i], Y[:i], sigma=sigma[:i], maxfev=1000 * (len(Y) + 1))
    init_ph = np.squeeze(init_ph)

    for ph in [init_ph, init_ph + np.pi]:
        # if signal is very long first fit roughly first 10 periods
        if (X[-1] - X[0]) * f0 > 10 and X.size > 20:
            i = int(10 / f0 / (X[1] - X[0]))

            # always fit to at most half the signal to avoid infinite recursion
            i = min(i, X.size // 2)

            popt, pcov, fit_details = fit_sine(
                X[:i],
                Y[:i],
                sigma=sigma[:i] if sigma is not None else None,
                model_key=model_key)

            if not any(np.isnan(popt)):
                f0 = popt[0]
                ph = popt[1]

        # actual fit
        k = Y.size // 2
        # check if signal is inverted to set initial decay constant signs
        inverted = max(Y[:k]) - min(Y[:k]) < max(Y[k:]) - min(Y[k:])
        sign = 1. if inverted else -1.

        if model_key in ['sine_model', 'sine_model_DC']:
            p0 = [f0, ph, (max(Y) - min(Y)) / 2]
        elif model_key in ['sine_unit_amplitude']:
            p0 = [f0, ph]
        else:
            p0 = [f0, ph, (max(Y) - min(Y)) / 2, sign / 100.]

        if model_key.split('_')[0] == 'double':
            p0 = [f0, ph, (max(Y) - min(Y)) / 4, sign / 10,
                  (max(Y) - min(Y)) / 4, sign / 100.]
            # p0.extend([(max(Y)-min(Y))/2, 0.0])

        if model_key[-2:] == 'DC':
            p0.append(np.average(Y))

        p0 = np.array(p0)

        # getting weird overflows, still fitting good
        np.seterr(over='ignore')

        # find point where phase is not correlated with phase
        if optimize_var_ph:
            p01 = p0.copy()

            def fun(t0):
                p01[1] = p0[1] + 2 * np.pi * f0 * t0

                try:
                    popt, pcov = scipy.optimize.curve_fit(
                        model, X - t0, Y,
                        sigma=sigma,
                        p0=p01,
                        maxfev=10000000)
                except RuntimeError:
                    # optimal parameters not found
                    return np.nan

                if pcov is np.inf:
                    # figure()
                    # errorbar(X, Y, yerr=sigma, fmt='k.', capsize=0, label='data')

                    # Tl = linspace(X[0], X[-1], 10000)
                    # plot(Tl, model(Tl - t0, *p01), label='fit start')
                    # plot(Tl, model(Tl - t0, *popt), label='fit end')
                    # legend(loc='best')
                    # title('Infinite covariance matrix!')
                    # show()

                    # return pcov
                    return np.nan
                else:
                    return pcov[1, 1]

            dt = (X[-1] - X[0]) / 2

            bracket = (X[0] - dt, (X[0] + X[-1]) / 2, X[-1] + dt)
            try:
                result = scipy.optimize.minimize_scalar(
                    fun=fun,
                    bracket=bracket,
                    method='Brent')
            except ValueError as er:
                # if fun(bracket[1]) > fun(bracket[0]) or
                # fun(bracket[1]) > fun(bracket[2])
                # then try downhill algorithm with just two elements in bracket

                bracket = (X[0] - dt, X[-1] + dt)
                result = scipy.optimize.minimize_scalar(
                    fun=fun,
                    bracket=bracket,
                    method='Brent')

            t0 = result.x
        else:
            t0 = 0.

        Xt0 = X - t0
        p0[1] = p0[1] + 2 * np.pi * f0 * t0

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",
                    category=scipy.optimize.OptimizeWarning)
                popt, pcov, infodict, errmsg, ier = scipy.optimize.curve_fit(
                    model, Xt0, Y,
                    sigma=sigma,
                    p0=p0,
                    # maxfev=10000000,
                    maxfev=10000,
                    full_output=True)
        except RuntimeError:
            # could not fit
            popt, pcov = [np.NaN] * len(p0), np.ones((len(p0), len(p0)))
            ier = 0

        np.seterr(over='print')

        if ier not in [1, 2, 3, 4]:
            # print('WARNING: sine fit did not converge:' + errmsg)
            fit_ok = False

        if pcov is np.ones((len(p0), len(p0))) * np.inf:
            fit_ok = False

        # print(errmsg)
        # try:
        #    popt, pcov = scipy.optimize.curve_fit(model, X, Y, sigma=sigma, p0=p0, maxfev=100*(len(Y)+1) )
        # except RuntimeError:
        # could not fit
        #    popt, pcov = [NaN]*4, ones( (len(p0), len(p0)) )
        fit = lambda t: model(t - t0, *popt)
        gfit = lambda t: model(t - t0, *p0)

        if not sigma_provided:
            # standard deviation estimation
            sigma *= np.sqrt(1 / (len(Y) - 1) * sum((fit(X) - Y) ** 2))

        chi2 = sum((fit(X) - Y) ** 2 / sigma ** 2)

        deg_of_freedom = len(Y) - len(popt) - 1
        reduced_chi2 = chi2 / deg_of_freedom
        CL = scipy.stats.chi2.cdf(chi2, deg_of_freedom)

        if reduced_chi2 > 1000:
            # print('WARNING: sine fit finished with red. chi^2 = {:.2g}'.format(
            #     reduced_chi2))
            fit_ok = False

        if fit_ok:
            break

    # plot
    # figure()
    # plot_ax = axes()
    if plot_ax is not None:
        if sigma_provided:
            plot_ax.errorbar(
                X, Y, yerr=sigma, fmt='k.', capsize=0, label='data')
        else:
            plot_ax.plot(X, Y, 'k.', label='data')
        Xf = np.linspace(X[0], X[-1], 10000)
        plot_ax.plot(Xf, fit(Xf),
                     label='fit (r.chi2: {:.2f},'
                           'f={:.2g}, ph={:.2g}(2pi))'.format(
            reduced_chi2, popt[0], popt[1] / 2 / np.pi))
        plot_ax.plot(Xf, gfit(Xf), 'r', alpha=0.5, label='fit start')
        if optimize_var_ph:
            plot_ax.axvline(t0, ls='--', lw=2, color='black', label='t0')
        # fitted_line1 = lambda x: popt[0] * exp(popt[1]*x) + popt[4]
        # fitted_line2 = lambda x: -popt[0] * exp(popt[1]*x) + popt[4]
        # plot(Xf, fitted_line1(Xf), 'g',
        #     alpha=0.5,
        # label='fit A ch. a={:.2g}+/-{:.2g}'.format(popt[0],
        #     sqrt(pcov[0,
        #     ])))
        # plot(Xf, fitted_line2(Xf), 'g', alpha=0.5)
        plot_ax.legend(loc='best')

    # show()

    # construct an object to hald details of the fit
    details = EstimationDetails()
    details.model_name = model_key
    details.model_function = model
    details.chi2 = chi2
    details.deg_of_freedom = deg_of_freedom
    details.reduced_chi2 = reduced_chi2
    details.CL = CL
    details.fit_ok = fit_ok
    details.popt = popt
    details.pcov = pcov
    details.t0 = t0

    if not fit_ok:
        return p0 * np.nan, np.ones([p0.size, p0.size]) * np.nan, details
    else:
        return popt, pcov, details


def window_fits(
        T, D, sD,
        iCrossings,
        model_key,
        verbose=False,
        plot_ax=None,
        invert_last_fit=False,
        throw_out_nans=True,
        optimize_var_ph=False,
        return_Pcov=False):
    Tf = []
    Dt = []
    Popt = []
    sPopt = []
    Pcov = []
    Ncrossings = []

    for i in range(len(iCrossings) - 1):
        i1 = iCrossings[i]
        i2 = iCrossings[i + 1]

        if invert_last_fit and i == len(iCrossings) - 2:
            Ti = -T[i1:i2 + 1][::-1] + T[i2]
            Di = D[i1:i2 + 1][::-1]
            sDi = sD[i1:i2 + 1][::-1]
        else:
            Ti = T[i1:i2] - T[i1]
            Di = D[i1:i2]
            sDi = sD[i1:i2]

        popt, pcov, details = fit_sine(
            X=Ti,
            Y=Di,
            sigma=sDi,
            model_key=model_key,
            optimize_var_ph=optimize_var_ph)

        t0 = details.t0

        if verbose:
            if plot_ax is not None:
                linspaceT = np.linspace(Ti[0], Ti[-1], 10000)
                fit = lambda t: models[model_key](t - t0, *popt)

                if invert_last_fit and i == len(iCrossings) - 2:
                    plot_ax.plot(linspaceT + T[i1], fit(linspaceT)[::-1])
                else:
                    plot_ax.plot(linspaceT + T[i1], fit(linspaceT))

                if optimize_var_ph:
                    plot_ax.axvline(T[i1] + t0, ls='--', color='black')

            print('    fit {:>4}/{}, {} points, Ampl = {:<7.6g} +/- {:.2g}'.format(
                i,
                iCrossings.size - 1,
                i2 - i1,
                popt[2],
                np.sqrt(pcov[2, 2])) if pcov is not np.inf else np.inf
                + ' ' * 20, end='\r')

        if optimize_var_ph:
            Tf.append(T[i1] + t0)
        else:
            Tf.append((T[i1] + T[i2 - 1]) / 2)
        Dt.append(T[i2] - T[i1])
        Popt.append(popt)
        sPopt.append(np.sqrt(abs(pcov.diagonal())))
        Pcov.append(pcov)

        Crossings = np.logical_and(D[i1 + 1:i2] > 0, D[i1:i2 - 1] < 0)
        Ncrossings.append(np.nonzero(Crossings)[0].size)

    Tf = np.array(Tf)
    Dt = np.array(Dt)
    Popt = np.array(Popt)
    sPopt = np.array(sPopt)
    Pcov = np.array(Pcov)
    Ncrossings = np.array(Ncrossings)
    F = Popt[:, 0]
    sF = Popt[:, 0]
    Ph = Popt[:, 1]
    sPh = Popt[:, 1]

    # if verbose:
    #     print(' '*80, end='\r')

    #     axarr[0].errorbar(T, D, yerr=sD, fmt='k.', capsize=0, label='data')
    if throw_out_nans:
        iNaNs = np.logical_or(
            np.logical_or(np.isnan(F), np.isnan(sF)),
            np.logical_or(np.isnan(Ph), np.isnan(sPh)))
        Tf = Tf[~iNaNs]
        Dt = Dt[~iNaNs]
        Popt = Popt[~iNaNs]
        sPopt = sPopt[~iNaNs]
        Ncrossings = Ncrossings[~iNaNs]

    if return_Pcov:
        return Tf, Dt, Popt, sPopt, Ncrossings, Pcov
    else:
        return Tf, Dt, Popt, sPopt, Ncrossings


def normalize_signal(T, D, sD, amplitude_method):
    D = D - np.average(D)

    # exp decay fit
    if amplitude_method == 'double_exp_decay_fit':
        M = np.copy(D)
        for i in range(len(M)):
            M[i] = max(abs(D[i:i + 30]))

        # plot( D, 'k.' )
        # plot( M, 'r-', lw=3 )

        two_exps = lambda x, a1, b1, a2, b2, c: a1 * \
            np.exp(b1 * x) + a2 * np.exp(b2 * x) + c
        p0 = [1, -1, 1, -1, np.average(D)]
        popt, pcov = scipy.optimize.curve_fit(two_exps, T, M, p0=p0)
        # print(popt)
        fit = lambda x: two_exps(x, *popt)
        sEst = np.sqrt(1 / (len(M) - 1) * sum((fit(T) - M) ** 2))
        chi2 = sum((fit(T) - M) ** 2 / sEst ** 2)
        deg_of_freedom = len(M) - len(popt) - 1
        reduced_chi2 = chi2 / deg_of_freedom

        Amplitudes = fit(T)
        sAmplitudes = sD

    elif amplitude_method == 'every_period_fit':
        iCrossings = divide_for_periods(D)

        Tf, Dt, Popt, sPopt, Ncrossings = window_fits(
            T, D, sD, iCrossings, 'sine_model', throw_out_nans=False)
        F = Popt[:, 0]
        sF = sPopt[:, 0]
        A = Popt[:, 2]
        sA = sPopt[:, 2]

        Amplitudes = np.zeros(D.size)
        sAmplitudes = np.zeros(D.size)

        Tfromfit = T.copy()
        Ffromfit = np.zeros(D.size)
        sFfromfit = np.zeros(D.size)
        for i in range(0, len(iCrossings) - 1):
            i1 = iCrossings[i]
            i2 = iCrossings[i + 1]
            Amplitudes[i1:i2] = A[i]
            sAmplitudes[i1:i2] = sA[i]
            Ffromfit[i1:i2] = F[i]
            sFfromfit[i1:i2] = sF[i]

    elif amplitude_method == 'every_period_max':
        # fit amplitude for every perdiod
        # if verbose:
        #     print('Fitting amplitude for every period:')
        iCrossings = divide_for_periods(D)

        # if verbose:
        #     for i in iCrossings:
        #         axarr[0].axvline(T[i], color='blue', alpha=0.2)
        # prealocate memory
        Amplitudes = np.zeros(D.size)
        sAmplitudes = np.zeros(D.size)

        Tfromfit = T.copy()
        Ffromfit = np.zeros(D.size)
        sFfromfit = np.zeros(D.size)

        for i in range(len(iCrossings) - 1):
            i1 = iCrossings[i]
            i2 = iCrossings[i + 1]

            Di = D[i1:i2]

            # simple maximum
            Amplitudes[i1:i2] = max(abs(Di))
            sAmplitudes[i1:i2] = sD[i1]

    # if verbose:
    #     print(' '*80, end='\r')

    # ignore part of the data
    Cond = np.logical_and(Amplitudes > 0, D <= Amplitudes)

    Amplitudes = Amplitudes[Cond]
    sAmplitudes = sAmplitudes[Cond]
    D = D[Cond]
    sD = sD[Cond]
    T = T[Cond]

    # if verbose:
    #     axarr[0].errorbar(T, D, yerr=sD, fmt='k.', capsize=0, label='data')
    # axarr[0].plot(T, M, 'r.', label='data moving max')
    # axarr[0].plot(T, fit(T), 'b-', lw=3, label='fit, red. chi2={:.2f}'.format(reduced_chi2))
    #     axarr[0].legend()
    #     axarr[0].errorbar(T, Amplitudes, yerr=sAmplitudes, fmt='r-', capsize=0)
    #     axarr[0].errorbar(T, -Amplitudes, yerr=sAmplitudes, fmt='r-', capsize=0)

    sD = np.sqrt(
        (sD / Amplitudes) ** 2 + (sAmplitudes * D / Amplitudes ** 2) ** 2)
    D /= Amplitudes

    return T, D, sD


def divide_for_periods(D):
    """For given sine-like signal D return array of indices where it can be
    subdivided into periods.
    """
    Crossings = np.logical_and(D[1:] > 0, D[:-1] < 0)
    iCrossings = np.nonzero(Crossings)[0]
    iCrossings = np.resize(
        iCrossings,
        iCrossings.size - np.remainder(iCrossings.size, 2))

    return iCrossings


def total_phase(T, D, sD, ph1, ph2, i1, i2, fs):
    """
    Give total phase difference (accumulated) between point D[i1] and D[i2],
    where ph1/ph2 are phases fitted in points D[i1]/D[i2]

    FIXED SAMPLING RATE
    """
    if i2 - i1 < 2 and ph2 - ph1 < 0.00001:
        return 0., np.array([])

    ph1r = np.remainder(ph1, 2*np.pi)
    ph2r = np.remainder(ph2, 2*np.pi)

    A = D[i1:i2+1]
    sA = sD[i1:i2+1]
    Ta = T[i1:i2+1]

    Crossings = np.logical_and(A[1:]>0, A[:-1]<0)

    # frequency estimation
    FFT = np.absolute(np.fft.fft(D))
    FFT = FFT[ : len(FFT)//2 ]
    FFT[0] = 0.0
    freqs = np.fft.fftfreq(D.size, 1/fs)[ : len(FFT) ]
    f0 = abs(freqs[np.argmax(FFT)])

    n = int(fs / f0 * 0.75)

    # delete these crossings that are closer to other crossing than 0.75 a period
    for i in range(0, len(Crossings)):
        if Crossings[i]:
            Crossings[i+1 : i+n] = False


    n = int(np.ceil(fs / f0 * 0.5))

    # correct for weird cases
    if ph1r > 0.75 * 2*np.pi and not any(Crossings[:n]):
        Crossings[0] = True

    if ph1r < 0.25 * 2*np.pi and any(Crossings[:n]):
        Crossings[:n] = np.r_[[False] * Crossings[:n]]

    if ph2r < 0.25 * 2*np.pi and not any(Crossings[-n:]):
        Crossings[-1] = True

    if ph2r > 0.75 * 2*np.pi and any(Crossings[-n:]):
        Crossings[-n:] = np.r_[[False] * Crossings[-n:].size]

    Ncrossings = np.nonzero(Crossings)[0].size
    Crossings_ts = ((Ta[1:] + Ta[:-1]) / 2)[Crossings]

    total_ph = (2*np.pi - ph1r) + (Ncrossings-1)*2*np.pi + ph2r
    # print(sin(ph1r), sin(ph2r))
    # print(D[i1], D[i2])

    # figure()
    # errorbar(Ta, A, yerr=sA, fmt='k.', label='phases %.2g %.2g' % (ph1r, ph2r))
    # for t in Crossings_ts:
    #     axvline(t, color='red')
    # legend(loc='best')
    # show()

    return total_ph, Crossings_ts


def total_phase_t(T, D, sD, ph1, ph2, t1, t2, fs):
    try:
        i1 = np.argwhere(T<=t1)[-1,0]
    except IndexError:
        i1 = 0

    try:
        i2 = np.argwhere(T>=t2)[0,0]
    except IndexError:
        i2 = T.size - 1

    return total_phase(T, D, sD, ph1, ph2, i1, i2, fs)


def cum_phase_t(T, D, sD, ph1, ph2, t1, t2, fs):
    total_ph_diff, Crossings_ts = total_phase_t(T, D, sD, ph1, ph2, t1, t2, fs)

    ph1r = np.remainder(ph1, 2*np.pi)

    # figure()
    # ph2r = remainder(ph2, 2*pi)
    # errorbar(T, D, yerr=sD, fmt='k.', label='phases %.2g %.2g' % (ph1r, ph2r))
    # for t in Crossings_ts:
    #     axvline(t, color='red')
    # legend(loc='best')
    # axvline(t1, color='blue')
    # axvline(t2, color='blue')

    return ph1r + total_ph_diff


def direct_fit(T, D, sD, model_key, plot_ax=None):
    """
    Directly fit the whole signal in order to estimate its frequency.

    This method simply fits the whole signal. If double_exp is False then
    fit model is an exponentially damped sine, if True - damping is a sum of
    two exponents.

    Parameters
    ----------
    T : array
        The array of times where the signal is sampled.
    D : array
        The samples of the signal.
    sD : array
        The array of the uncertainties (standard-deviation estimates)
        of the samples, used to determine weight for the fit.
    model_key : str
        Has to refer to one of the built-in models define in the
        dictionary `pyfid.estimation.models`.
    plot_ax : matplotlib.Axes, optional
        If passed the result of the fit will be plotted there.

    Returns
    -------
    f : float
        The estimator of the average frequency of the signal.
    sf : float
        The estimator of the uncertainty of the average frequency estimate.
    details: EstimationDetails
        On object holding the details of the estimation. `details.fit_details`
        contains further details from the underlying
        `pyfid.estimation.fit_sine` function.
    """
    popt, pcov, fit_details = fit_sine(
        T - T[0], D,
        sigma=sD,
        plot_ax=plot_ax,
        model_key=model_key)

    f = popt[0]
    sf = np.sqrt(pcov[0, 0])

    details = EstimationDetails()

    details.fit_details = fit_details
    details.f = f
    details.sf = sf

    return f, sf, details


def two_windows(
        T, D, sD,
        submethod='phase',
        prenormalize=False,
        double_exp=(True, False),
        phase_at_end=True,
        win_len=15.0,
        plot_fig=None,
        verbose=False):
    """
    TODO

    Parameters
    ----------
    TODO
    """
    try:
        ls1, ls2 = win_len  # s
    except TypeError:
        ls1, ls2 = win_len, win_len

    t_start = T[0]
    t_end = T[-1]
    fs = 1 / np.mean(np.diff(T))

    # Crop the window size, preserving the ratio of the windows' sizes.
    if ls1 + ls2 > t_end - t_start:
        ratio = ls1 / ls2
        duration = t_end - t_start
        ls2 = duration / (1 + ratio)
        ls1 = duration - ls2

    t_win1 = t_start + ls1
    t_win2 = t_end - ls2

    if prenormalize:
        # model_key = 'sine_model'
        model_key = 'sine_unit_amplitude'
        Torig = T.copy()
        Dorig = D.copy() - np.average(D)
        T, D, sD = normalize_signal(T, D, sD,
            amplitude_method='every_period_fit')
    else:
        model_key = 'damped_sine'
        D = D - np.average(D)

    l1 = np.argwhere(T > t_win1)[0, 0]
    l2 = T.size - 1 - np.argwhere(T < t_win2).flatten()[-1]
    tl2 = T[-l2]

    if plot_fig is not None:
        gs = plot_fig.add_gridspec(2, 2, height_ratios=[1, 1])
        ax = plot_fig.add_subplot(gs[0, :])
        ax.errorbar(T, D, yerr=sD, fmt='k.', capsize=0)
        ax.axvspan(T[0], T[l1], alpha=0.3)
        ax.axvspan(tl2, T[-1], alpha=0.3)

        print('Fit to first {} s with model "{}"'.format(ls1, model_key))

    popt, pcov, fit_details1 = fit_sine(
        T[:l1] - T[0],
        D[:l1],
        sigma=sD[:l1],
        plot_ax=plot_fig.add_subplot(gs[1, 0]) if plot_fig is not None else None,
        model_key='double_damped_sine' if double_exp[0] else model_key)

    f1 = popt[0]
    with np.errstate(invalid="ignore"):
        # sometimes the covariance of the parameters cannot be estimated
        sf1 = np.sqrt(pcov[0, 0])

    ph1 = np.remainder(popt[1], 2 * np.pi)
    with np.errstate(invalid="ignore"):
        # sometimes the covariance of the parameters cannot be estimated
        sph1 = np.sqrt(pcov[1, 1])

    # construct an object to hold details about the estimation
    details = EstimationDetails()

    details.first_window_details = fit_details1
    details.f1 = f1
    details.sf1 = sf1
    details.ph1 = ph1
    details.sph1 = sph1

    # Fit in the second window

    if verbose:
        print('    red. chi^2 = {:.2g}'.format(fit_details1.reduced_chi2))
        print('    f1 = {} +/- {:.2g}'.format(f1, sf1))
        print(
            '    ph1 = {} +/- {:.2g} (2pi)'.format(ph1 / 2 / np.pi, sph1 / 2 / np.pi))

        print('Fit to last {} s with model "{}"'.format(ls2, model_key))

    if phase_at_end:
        # invert last fit
        D2 = D[-l2:][::-1]
        sD2 = sD[-l2:][::-1]
        T2 = -T[-l2:][::-1] + T[-1]
    else:
        T2 = T[-l2:] - T[-l2]
        D2 = D[-l2:]
        sD2 = sD[-l2:]

    popt, pcov, fit_details2 = fit_sine(
        T2,
        D2,
        sigma=sD2,
        plot_ax=plot_fig.add_subplot(gs[1, 1]) if plot_fig is not None else None,
        model_key='double_damped_sine' if double_exp[1] else model_key)

    f2 = popt[0]
    sf2 = np.sqrt(pcov[0, 0])

    ph2 = np.remainder(popt[1], 2 * np.pi)
    if phase_at_end:
        ph2 = np.pi - ph2
    sph2 = np.sqrt(pcov[1, 1])

    details.second_window_details = fit_details2
    details.f2 = f2
    details.sf2 = sf2
    details.ph2 = ph2
    details.sph2 = sph2

    if verbose:
        print('    red. chi^2 = {:.2g}'.format(fit_details2.reduced_chi2))
        print('    f2 = {} +/- {:.2g}'.format(f2, sf2))
        print(
            '    ph2 = {} +/- {:.2g} (2pi)'.format(ph2 / 2 / np.pi, sph2 / 2 / np.pi))

        print('Phase method:')

    # crossings
    # trick to always get correct number of zero-crossings
    Tb = Torig if prenormalize else T
    total_ph, Crossings_ts = total_phase(
        T=Tb,
        D=Dorig if prenormalize else D,
        sD=sD,
        ph1=ph1,
        ph2=ph2,
        i1=np.argwhere(Tb == T[0]).flatten()[0],
        i2=np.argwhere(Tb == (T[-1] if phase_at_end else tl2)).flatten()[0],
        fs=fs)

    if phase_at_end:
        f = total_ph / (T[-1] - T[0]) / (2 * np.pi)
        sf = np.sqrt(sph2 ** 2 + sph1 ** 2) / (T[-1] - T[0]) / (2 * np.pi)
    else:
        f = total_ph / (T[-l2] - T[0]) / (2 * np.pi)
        sf = np.sqrt(sph2 ** 2 + sph1 ** 2) / (T[-l2] - T[0]) / (2 * np.pi)

    details.number_of_crossings = len(Crossings_ts)
    details.total_phase_difference = total_ph / 2 / np.pi
    details.f = f
    details.sf = sf

    if verbose:
        print('    Number of crossings: {}'.format(details.number_of_crossings))
        print('    Total phase diff.: {} (2pi)'.format(total_ph / 2 / np.pi))
        print('    f = {} +/- {:.2g}'.format(f, sf))

        for t in Crossings_ts:
            ax.axvline(t + 0.5 / fs, color='red', alpha=0.3)
        plot_fig.set_tight_layout(True)
        plot_fig.show()

    # wrong arrays for phase at end!!!!
    return f, sf, details
