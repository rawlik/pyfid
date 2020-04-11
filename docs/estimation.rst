Estimation
==========

Introduction
------------
The package includes many methods to estimate the frequency.

.. autoclass:: pyfid.estimation.EstimationDetails


Direct fitting
--------------
The underlying heavy lifting is done by the `pyfid.estimation.fit_sine` function.

.. autofunction:: pyfid.estimation.fit_sine

This function can use one of many models for fitting:

.. literalinclude:: ../pyfid/estimation.py
    :language: python
    :lines: 13-37


Directly fit whole signal
-------------------------
.. autofunction:: pyfid.estimation.direct_fit

.. figure:: ../examples/output/direct_fit.png

.. literalinclude:: ../examples/direct_fit.py
   :language: python


The two windows methods
-----------------------
In this method a fit is performed in the first part of the signal and the last.
From each the phase is estimated, and the difference is used to estimate
the average frequency.

TODO References

.. autofunction:: pyfid.estimation.two_windows



TODO Others
-----------
.. automodule:: pyfid.estimation
    :exclude-members: fit_sine, EstimationDetails, direct_fit, two_windows, window_fits, normalize_signal, divide_for_periods, total_phase, total_phase_t, cum_phase_t
    :members:
    :undoc-members:
    :member-order: bysource


Misc
----
.. autofunction:: pyfid.estimation.window_fits

.. autofunction:: pyfid.estimation.normalize_signal

.. autofunction:: pyfid.estimation.divide_for_periods

.. autofunction:: pyfid.estimation.total_phase

.. autofunction:: pyfid.estimation.total_phase_t

.. autofunction:: pyfid.estimation.cum_phase_t
