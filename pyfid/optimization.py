import numpy as np
import scipy.stats


def std_CL(A, two_bounds=False, ignore_nans=False):
    """Calculate standard deviation estimate with bound for
    confidence level 1 sigma.

    TODO
    """
    cl = scipy.stats.norm.cdf(1) - scipy.stats.norm.cdf(-1)

    S = np.array(A)

    if ignore_nans:
        S = S[~np.isnan(S)]

    stdS = np.std(S, ddof=1)
    # degrees of freedom
    df = S.size - 1

    try:
        S2 = 1 / df * sum((S - np.average(S)) ** 2)

        low = np.sqrt(df * S2 / scipy.stats.chi2.ppf(0.5 + cl / 2, df))
        up = np.sqrt(df * S2 / scipy.stats.chi2.ppf(0.5 - cl / 2, df))
    except ZeroDivisionError:
        low = 0
        up = 0

    dlow = stdS - low
    dup = up - stdS

    if two_bounds:
        return stdS, (dlow, dup)
    else:
        return stdS, (dlow + dup) / 2


def average_CL(A, sA=None, ignore_nans=False):
    """Calculate average with bound for
    confidence level 1 sigma.

    TODO
    """
    S = np.array(A)

    if ignore_nans:
        if sA is None:
            S = S[~np.isnan(S)]
        else:
            mask = np.logical_or(np.isnan(S), np.isnan(sA))
            S = S[~mask]

    avgS = np.average(S)

    if sA is None:
        savgS = np.std(S, ddof=1) / np.sqrt(S.size)
    else:
        if len(A) != len(sA):
            raise Exception('A and sA have different sizes')

        sS = np.array(sA)

        if ignore_nans:
            sS = sS[~mask]

        savgS = np.sqrt(sum(sS ** 2)) / S.size

    return avgS, savgS


def accuracy_and_precision(sim, estimator, nsignals, full_output=False):
    """
    TODO
    """
    accuracies = []
    precisions = []

    signals = sim.simulate(nsignals)

    result_generator = ( estimator(sim.T, signal, sim.sigma())
        for signal in signals )

    # the actual calculation happens here
    f, sf, details = zip(*result_generator)

    accuracy, saccuracy = average_CL(np.array(f) - sim.real_favg(), ignore_nans=True)
    precision, sprecision = std_CL(np.array(f), ignore_nans=True)

    if full_output:
        return accuracy, precision, saccuracy, sprecision
    else:
        return accuracy, precision


def accuracy_and_precision_different_sims(sim_gen, estimator, nsimulations,
        nsignals, full_output=False):
    """
    A loop, new simulation generated, for each simulation many signals
    are generated, so that the precision (the standard deviation of
    frequency estimates) and accuracy (the average deviation from the true
    average frequency for that simulation) can be estimated.

    Parameters
    ----------
    sim_gen : callable
        A function that will return an `FIDsim` object when called.

    TODO
    """
    results = []

    for _isimulation in range(nsimulations):
        sim = sim_gen()

        accuracy, precision, saccuracy, sprecision = accuracy_and_precision(
            sim=sim, estimator=estimator, nsignals=nsignals, full_output=True)

        results.append((accuracy, precision, saccuracy, sprecision))

    results = np.array(results)

    accuracy, saccuracy = average_CL(results[:, 0], results[:, 2], ignore_nans=True)
    precision, sprecision = average_CL(results[:, 1], results[:, 3], ignore_nans=True)

    if full_output:
        return accuracy, precision, saccuracy, sprecision
    else:
        return accuracy, precision
