What is wave reconstruction and propagation? (WRP)
==================================================

Most simply, wave reconstruction and propagation means taking measurements of wave heights at one location and turning
them into the phase resolved wave prediction at another location. 

In :ref:`layout` one can see a physical wave tank in which we generate waves that travel in a single direction.
We measure waves at one location, and predict their shape further down the tank where a float sits on the surface. 

.. _layout:
.. image:: ../wrp-layout.png

General technique
-----------------

The general approach is to deconstruct the entire wave field into a linear superposition of many constituent waves. 
Using linear wave theory, the speed of each constituent wave is known based the dispersion relation for water surface waves. 
With the combined knowledge of the constituent waves present and their speeds, the wave field can be propagated to a new location and time 
of interest.

To see the math which supports this framework, please read the :doc:`theory` page.