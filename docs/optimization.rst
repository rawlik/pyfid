Optimization
============


Accuracy and precision
----------------------
A function that evaluates the accuracy and precision of an estimation method
for a single particular phase evalution:

.. autofunction:: pyfid.optimization.accuracy_and_precision

And a one that does averages this for multiple phase evolutions. For that
a fuction that generates the simulation has to be provided.

.. autofunction:: pyfid.optimization.accuracy_and_precision_different_sims


Parameter optimization
----------------------
In the following example an estimation method - the two windows method - the
size of the second window is parametrised. For each parameter value the
accuracy and precision are evaluated, averaged over many random drifts.

.. autofunction:: pyfid.optimization.bisect_parameter

.. figure:: ../examples/output/two_windows_scan.png

.. literalinclude:: ../examples/two_windows_scan.py
   :language: python


Optimizing the window size
--------------------------
TODO


Misc
----
.. autofunction:: pyfid.optimization.std_CL
.. autofunction:: pyfid.optimization.average_CL

