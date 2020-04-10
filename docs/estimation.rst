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


TODO Others
-----------

.. automodule:: pyfid.estimation
    :exclude-members: fit_sine, EstimationDetails
    :members:
    :undoc-members:
    :member-order: bysource
