=======================
compact_radial_basis.py
=======================

- Interpolates data using compactly supported radial basis functions of minimal degree and sparse matrix algebra [Buhmann2003]_ [Wendland1995]_ [Wendland2005]_

Calling Sequence
################

.. code-block:: python

    import spatial_interpolators as spi
    ZI = spi.compact_radial_basis(xs, ys, zs, XI, YI, dimension, order, method='wendland')

`Source code`__

.. __: https://github.com/tsutterley/spatial-interpolators/blob/master/spatial_interpolators/compact_radial_basis.py


Arguments
#########

1. ``xs``: input x-coordinates
2. ``ys``: input y-coordinates
3. ``zs``: input data
4. ``XI``: output x-coordinates for data grid
5. ``YI``: output y-coordinates for data grid
6. ``dimension``: spatial dimension of Wendland function
7. ``order``: smoothness order of Wendland function

Keyword arguments
#################

- ``smooth``: smoothing weights
- ``radius``: scaling factor for the basis function
- ``method``: compactly supported radial basis function

    * ``'wendland'``

Returns
#######

- ``ZI``: interpolated data grid

References
##########

.. [Buhmann2003] M. Buhmann, "Radial Basis Functions", *Cambridge Monographs on Applied and Computational Mathematics*, (2003).

.. [Wendland1995] H. Wendland, "Piecewise polynomial, positive definite and compactly supported radial functions of minimal degree," *Advances in Computational Mathematics*, 4, 389--396, (1995). `doi: 10.1007/BF02123482 <https://doi.org/10.1007/BF02123482>`_

.. [Wendland2005] H. Wendland, "Scattered Data Approximation", *Cambridge Monographs on Applied and Computational Mathematics*, (2005).
