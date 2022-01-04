===================
sph_radial_basis.py
===================

- Interpolates data over a sphere using radial basis functions
- QR factorization option to eliminate ill-conditioning [Fornsberg2007]_ [Fornsberg2011]_
-
Calling Sequence
################

.. code-block:: python

    import spatial_interpolators as spi
    output = spi.sph_radial_basis(lon, lat, data, longitude, latitude, method='inverse')

`Source code`__

.. __: https://github.com/tsutterley/spatial-interpolators/blob/master/spatial_interpolators/sph_radial_basis.py


Arguments
#########

1. ``lon``: input longitude
2. ``lat``: input latitude
3. ``data``: input data
4. ``longitude``: output longitude
5. ``latitude``: output latitude

Keyword arguments
#################

- ``smooth``: smoothing weights
- ``epsilon``: adjustable constant for distance functions
- ``method``: radial basis function

    * ``'multiquadric'`` [#f1]_
    * ``'inverse_multiquadric'`` [#f1]_ or ``'inverse'`` [#f1]_
    * ``'inverse_quadratic'`` [#f1]_
    * ``'gaussian'`` [#f1]_
    * ``'linear'``
    * ``'cubic'``
    * ``'quintic'``
    * ``'thin_plate'``
- ``QR``: use QR factorization algorithm of [Fornsberg2007]_
- ``norm``: distance function for radial basis functions (if not using QR)

    * ``'euclidean'``: Euclidean Distance with distance_matrix
    * ``'GCD'``: Great-Circle Distance using n-vectors with angle_matrix

.. [#f1] has option for QR factorization method

Returns
#######

- ``output``: interpolated data

References
##########

.. [Fornsberg2007] B. Fornberg and C. Piret, "A stable algorithm for flat radial basis functions on a sphere," *SIAM Journal on Scientific Computing*, 30(1), 60--80, (2007). `doi: 10.1137/060671991 <https://doi.org/10.1137/060671991>`_

.. [Fornsberg2011] B. Fornberg, E. Larsson, and N. Flyer, "Stable Computations with Gaussian Radial Basis Functions," *SIAM Journal on Scientific Computing*, 33(2), 869--892, (2011). `doi: 10.1137/09076756X <https://doi.org/10.1137/09076756X>`_
