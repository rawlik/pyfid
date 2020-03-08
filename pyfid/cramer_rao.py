import pylab as pl
from scipy.misc import derivative
import inspect
from scipy.optimize import curve_fit


def cramer_rao(model, p0, X, noise, show_plot=False, quad_precision=True):
    """Calculate inverse of the Fisher information matrix for model
    sampled on grid X with parameters p0. Assumes samples are not
    correlated and have equal variance noise^2.

    Parameters
    ----------
    model : callable
        The model function, f(x, ...).  It must take the independent
        variable as the first argument and the parameters as separate
        remaining arguments.
    X : array
        Grid where model is sampled.
    p0 : M-length sequence
        Point in parameter space where Fisher information matrix is
        evaluated.
    noise: scalar
        Squared variance of the noise in data.
    show_plot : boolean
        If True shows plot.

    Returns
    -------
    iI : 2d array
        Inverse of Fisher information matrix.
    """
    # increase precission
    if quad_precision:
        p0l = pl.array(p0).astype(pl.longdouble)
        Xl = X.astype(pl.longdouble)
        noisel = pl.longdouble(noise)
    else:
        p0l = pl.array(p0)
        Xl = X
        noisel = noise

    argspec = inspect.getargspec(model)
    labels = argspec.args[1:]

    if argspec.varargs is not None:
        for i in range(len(p0l) - len(labels)):
            labels.append('%s%d' % (argspec.varargs[0], i))

    p0dict = dict(zip(labels, p0l))

    D = pl.zeros((len(p0l), Xl.size))
    for i, argname in enumerate(labels):
        def func(x, p):
            current_p_list = p0l.copy()
            current_p_list[i] = p
            return model(x, *current_p_list)

        # use routine from numdifftools
        # https://code.google.com/p/numdifftools/
        # D[i,:] = [
        #     nd.Derivative(lambda p: func(x, p), order=4, romberg_terms=3, step_nom=p0dict[argname] * 1e-13)(p0dict[argname])
        #     for x in Xl ]

        # use routine from scipy
        D[i,:] = [
            derivative(
                lambda p: func(x, p),
                p0dict[argname],
                dx=(1e-14 if quad_precision else 1e-10) )
            for x in Xl ]

    if show_plot:
        pl.figure()
        pl.plot(Xl, model(Xl, *p0l), '--k', lw=2, label='signal')
        for d, label in zip(D, labels):
            pl.plot(Xl, d, '.-', label=label)
        pl.legend(loc='best')
        pl.title('Parameter dependence on particular data point')

    I = 1/noisel**2 * pl.einsum('mk,nk', D, D)
    iI = pl.inv(I)

    return iI.astype(pl.double)


def cramer_rao_monte_carlo(model, p0, X, noise, show_plot=False):
    Y = model(X, *p0)
    Y += noise * pl.randn(Y.size)

    popt, pcov = curve_fit(model, X, Y, p0=p0)

    if show_plot:
        pl.figure()
        pl.plot(X, model(X, *p0), 'b--', lw=2, label='signal')
        pl.plot(X, Y, '.k', lw=2, label='signal')
        pl.plot(X, model(X, *popt), 'r-', lw=2)
        pl.title('Parameter dependence on particular data point')

    return pcov


def cramer_rao_analytic(dt, T, tau1, tau2, sigma, A):
    raw_text = """
    (2*Sqrt(-((E**((dt*Floor(T/dt))/tau1)*\[Sigma]**2*Sinh(dt/tau1)**3*Sinh((dt*(1 + Floor(T/dt)))/tau1))/
     -        (A**2*dt**2*(Cosh((2*dt)/tau1) - Cosh((2*dt*(1 + Floor(T/dt)))/tau1) +
     -            2*Floor(T/dt)*(2 + Floor(T/dt))*Sinh(dt/tau1)**2)))))/Pi
    """
    parsed_text = raw_text \
        .replace('-  ','') \
        .replace('\[Sigma]', 'sigma') \
        .replace('E**(', 'pl.exp(') \
        .replace('Floor(', 'pl.floor(') \
        .replace('Sinh(', 'pl.sinh(') \
        .replace('Cosh(', 'pl.cosh(') \
        .replace('Pi','pl.pi') \
        .replace('Sqrt(', 'pl.sqrt(') \
        .replace('\n','')

    return eval(parsed_text)
