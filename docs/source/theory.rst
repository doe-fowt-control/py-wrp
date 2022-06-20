Theory
======


General technique
-----------------

The general approach is to deconstruct the entire wave field into a linear superposition of many constituent waves. 
Using linear wave theory, the speed of each constituent wave is known based the dispersion relation for water surface waves. 
With the combined knowledge of the constituent waves present and their speeds, the wave field can be propagated to a new location and time 
of interest.


Assumptions
___________

To attempt to represent the continuous surface with a set of frequencies, we assume that the spectrum can be reasonably represented 
by a finite frequency bandwidth, which agrees well with natural ocean wave spectrum that tend to have most of their energy around 
some peak frequency. This aligns with work done by Wu in 2004.

We also assume that deep water wave relationships hold, or that the depth is greater than half the wavelength. This allows us to 
introduce the dispersion relation for deep water waves, relating the horizontal wavenumber :math:`k` with frequency :math:`\omega` as 

.. math::
    \omega^2 = gk

where :math:`g` is the constant of gravitational acceleration.

Prediction zone
_______________

The spatiotemporal region where we expect a good match between the reconstruction and reality is called the 'prediction zone.' 
The prediction zone is defined by the amount of time used to measure the waves and the speed of the constituent waves. 

Figure :numref:`wu0` is borrowed from Wu 2004 which illustrates the procedure for calculating prediction zone from a single probe. 
For :math:`x_p` values greater than the position of the measurement probe :math:`x`, the time window for reasonable predictions at 
:math:`x_p` is based on the slowest and fastest group velocities, :math:`C_{gl}` and :math:`C_{gh}` respectively.

.. math::
    \frac{x_p - x}{C_{gl}} 
    \leq t 
    \leq T_a + \frac{x_p - x_}{C_{gh}}

.. _wu0:
.. figure:: images/wu-0.png
    :width: 600
    Prediction zone based on fastest and slowest group velocities, as well as assimilation time. Borrowed from Wu 2004.


Using multiple probes, the predictable region increases accordingly, where $x$ in the equation above becomes a reference to the largest and smallest locations in space.

.. math::
    \frac{ x_p - x_{\text{max}} } {C_{gl}} 
    \leq t 
    \leq T_a + \frac{ x_p - x_{\text{min}}}{C_{gh}}

.. _wu-ng:
.. figure:: images/wu-ng-pred.png
    :width: 600
    Prediction zone for multiple wave gauges. Borrowed from Wu 2004.

Spectral calculations
---------------------

Separately from reconstruction, a longer time scale is used to calculate the one directional frequency wave spectrum. By default, 30 seconds of wave height information is assimilated into the spectrum. We do this with the pwelch method to calculate power spectral density at the measurement gauges, taking the average for the case with multiple gauges. This method in MATLAB requires specification of three parameters: 


From the spectrum, we calculate the zeroth moment $m_0$ as the area under the spectral curve. The significant wave height is then found as

.. math::
    H_s = 4 * \sqrt{m_0}


Peak period :math:`T_p` is simply the period associated with the peak of the energy curve, where :math:`T = 2\pi / \omega`.

The fastest and slowest group velocities used for the prediction zone are also derived from the spectrum. To select the frequencies corresponding to these group velocities, we find frequencies that represent some fraction of the peak energy of the spectrum. This paper defaults to using a threshold parameter :math:`\mu = 0.05` (5% of the peak energy) but experimented with using up to 15\% cutoff. Desmars et. al. chose this approach, arguing that the asymptotic nature of wave spectrum tends to bring the higher selected frequency to be too high. 

For deep water, group velocities are related to cutoff frequencies by

.. math::
    c_g = \frac{1}{2}c = \frac{g}{2\omega}


Misfit indicator definition
---------------------------

To assess the accuracy of our wave prediction across multiple realizations we define the following misfit indicator. 

.. math::
    \epsilon(x, t) = \frac{1}{N_s}\sum_{i=1}^{N_s}|\eta(x,t) - \Tilde{\eta}(x,t)| / H_s

Where :math:`N_s` is the number of realizations, :math:`\eta` is the measured wave heights, :math:`\Tilde{\eta}` is the predicted wave heights,
and :math:`H_s` is the significant wave height calculated from the spectral data. For our experiments, multiple realizations were drawn from a 
single continuous wave generation by isolating data from completely different points in time. 
