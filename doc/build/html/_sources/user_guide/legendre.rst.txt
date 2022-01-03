===========
legendre.py
===========

- Computes associated Legendre functions of degree ``l`` evaluated for elements ``x`` [Abramowitz1965]_ [Jacobs1987]_
- ``l`` must be a scalar integer and ``x`` must contain real values ranging -1 <= ``x`` <= 1
- Unnormalized associated Legendre function values will overflow for ``l`` > 150

Calling Sequence
################

.. code-block:: python

    import spatial_interpolators as spi
    Pl = spi.legendre(l, x)

`Source code`__

.. __: https://github.com/tsutterley/spatial-interpolators/blob/master/spatial_interpolators/legendre.py


Arguments
#########

- ``l``: degree of Legrendre polynomials
- ``x``: elements ranging from -1 to 1. Typically ``cos(theta)``, where ``theta`` is the colatitude in radians

Keyword arguments
#################

- ``NORMALIZE``: output Fully Normalized Associated Legendre Functions

Returns
#######

- ``Pl``: Legendre polynomials of degree ``l``

References
##########

.. [Abramowitz1965] M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, 1046 pp., (1965).

.. [Jacobs1987] J. A. Jacobs, *Geomagnetism*, Volume 1, 1st Edition, 832 pp., (1987).
