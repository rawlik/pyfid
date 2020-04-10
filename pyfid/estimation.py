"""
The module `estimation` holds various methods of estimating the frequency
of an FID signal.
"""
import inspect

import numpy as np
import scipy.optimize
import scipy.stats

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


def fit_sine(X, Y, sigma=None, plot_ax=None, model_key='damped_sine_DC',
             optimize_var_ph=False):
    # check if size of data is larger than number of model parameters
    nparms = len(inspect.getargspec(models[model_key])[0]) - 1
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


def direct_fit(
        T, D, sD,
        double_exp=True,
        verbose=False):
    """
    Fit the whole signal.

    This method simply fits the whole signal. If double_exp is False then
    fit model is an exponentially damped sine, if True - damping is a sum of
    two exponents.
    """

    if verbose:
        fig = figure()
        ax = fig.add_subplot(111)
        ax.errorbar(T, D, yerr=sD, fmt='k.', capsize=0)

    popt, pcov, fit_details = fit_sine(
        T - T[0], D,
        sigma=sD,
        plot_ax=ax if verbose else None,
        model_key='double_damped_sine_DC' if double_exp else 'damped_sine_DC')

    f = popt[0]
    sf = np.sqrt(pcov[0, 0])

    details = EstimationDetails()

    details.fit_details = fit_details
    details.f = f
    details.sf = sf

    return f, sf, details
