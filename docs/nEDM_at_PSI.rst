nEDM-at-PSI
===========

Parameters of the FID
---------------------
The parameters are stored in the `pyfid.nEDMatPSI` module.

.. automodule:: pyfid.nEDMatPSI
   :exclude-members: optimize_window_size
   :members:
   :undoc-members:
   :member-order: bysource


The nEDM-at-PSI experiment filter
---------------------------------
This is an implementation of the particular filter used in the nEDM-at-PSI
experiment.

.. figure:: ../examples/output/filter_frequency_response.png
.. figure:: ../examples/output/filter_noise_filtering.png
.. figure:: ../examples/output/filter_signal_filtering.png
.. figure:: ../examples/output/filter_visualisation.png

.. literalinclude:: ../examples/nEDM_at_PSI_filter.py
   :language: python


Estimating the amplitude of noise from a signal
-----------------------------------------------
.. figure:: ../examples/output/nEDM_at_PSI_noise_estimation.png

.. literalinclude:: ../examples/nEDM_at_PSI_noise_estimation.py
   :language: python


Optimizing the window size
--------------------------
.. autofunction:: pyfid.nEDMatPSI.optimize_window_size
