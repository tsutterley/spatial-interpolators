=============
sph_spline.py
=============

- Interpolates data over a sphere using spherical surface splines in tension [Wessel2008]_

Calling Sequence
################

.. code-block:: python

    import spatial_interpolators as spi
    output = spi.sph_spline(lon, lat, data, longitude, latitude, tension=0.5)

`Source code`__

.. __: https://github.com/tsutterley/spatial-interpolators/blob/master/spatial_interpolators/sph_spline.py


Arguments
#########

1. ``lon``: input longitude
2. ``lat``: input latitude
3. ``data``: input data
4. ``longitude``: output longitude
5. ``latitude``: output latitude

Keyword arguments
#################

- ``tension``: tension to use in interpolation

Returns
#######

- ``output``: interpolated data

References
##########

.. [Wessel2008] P. Wessel, and J. M. Becker, "Interpolation using a generalized Green's function for a spherical surface spline in tension," *Geophysical Journal International*, 174(1), 21--28, (2008). `doi: 10.1111/j.1365-246X.2008.03829.x <https://doi.org/10.1111/j.1365-246X.2008.03829.x>`_
