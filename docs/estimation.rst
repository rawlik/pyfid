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


Direct fit
----------
.. autofunction:: pyfid.estimation.direct_fit

TODO Others
-----------

.. automodule:: pyfid.estimation
    :exclude-members: fit_sine, EstimationDetails
    :members:
    :undoc-members:
    :member-order: bysource