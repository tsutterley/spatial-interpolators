===================
sph_radial_basis.py
===================

- Interpolates data over a sphere using radial basis functions
- QR factorization option to eliminate ill-conditioning

Calling Sequence
################

.. code-block:: python

    import spatial_interpolators as spi
    output = spi.sph_radial_basis(lon, lat, data, longitude, latitude, method='inverse')

`Source code`__

.. __: https://github.com/tsutterley/spatial-interpolators/blob/master/spatial_interpolators/sph_radial_basis.py

.. autofunction:: spatial_interpolators.sph_radial_basis