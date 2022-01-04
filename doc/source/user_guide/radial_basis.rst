===============
radial_basis.py
===============

- Interpolates data using radial basis functions [Hardy1971]_ [Buhmann2003]_

Calling Sequence
################

.. code-block:: python

    import spatial_interpolators as spi
    ZI = spi.radial_basis(xs, ys, zs, XI, YI, method='inverse')

`Source code`__

.. __: https://github.com/tsutterley/spatial-interpolators/blob/master/spatial_interpolators/radial_basis.py


Arguments
#########

1. ``xs``: input x-coordinates
2. ``ys``: input y-coordinates
3. ``zs``: input data
4. ``XI``: output x-coordinates for data grid
5. ``YI``: output y-coordinates for data grid

Keyword arguments
#################

- ``smooth``: smoothing weights
- ``metric``: distance metric to use (default euclidean)
- ``epsilon``: adjustable constant for distance functions
- ``method``: radial basis function

    * ``'multiquadric'``
    * ``'inverse_multiquadric'`` or ``'inverse'``
    * ``'inverse_quadratic'``
    * ``'gaussian'``
    * ``'linear'``
    * ``'cubic'``
    * ``'quintic'``
    * ``'thin_plate'``
- ``polynomial``: polynomial order if augmenting radial basis functions


Returns
#######

- ``ZI``: interpolated data grid

References
##########

.. [Hardy1971] R. L. Hardy, "Multiquadric equations of topography and other irregular surfaces," *Journal of Geophysical Research*, 76(8), 1905-1915, (1971). `doi: 10.1029/JB076i008p01905 <https://doi.org/10.1029/JB076i008p01905>`_

.. [Buhmann2003] M. Buhmann, "Radial Basis Functions", *Cambridge Monographs on Applied and Computational Mathematics*, (2003).