Estimation
==========

Introduction
------------
The package is structured in a way to provide different methods of estimating
the average frequency of an FID signal.


Fitting an arbitrary oscillating signal
---------------------------------------
The underlying heavy lifting is done by the `pyfid.estimation.fit_sine` function.

.. autofunction:: pyfid.estimation.fit_sine

This function can use one of many models for fitting:

.. literalinclude:: ../pyfid/estimation.py
    :language: python
    :lines: 13-37

It returns, like all estimation methods in this module, an `EstimationDetails`
object, which holds various, often method-specific information about the
estimation.

.. autoclass:: pyfid.estimation.EstimationDetails


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

.. figure:: ../examples/output/two_windows.png

.. literalinclude:: ../examples/two_windows.py
   :language: python


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
