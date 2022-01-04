===================
barnes_objective.py
===================

- Optimally interpolates data using Barnes objective analysis using a successive corrections scheme [Barnes1994a]_ [Barnes1994b]_ [Barnes1994c]_ [Daley1991]_

Calling Sequence
################

.. code-block:: python

    import spatial_interpolators as spi
    ZI = spi.barnes_objective(xs, ys, zs, XI, YI, XR, YR)

`Source code`__

.. __: https://github.com/tsutterley/spatial-interpolators/blob/master/spatial_interpolators/barnes_objective.py


Arguments
#########

1. ``xs``: input x-coordinates
2. ``ys``: input y-coordinates
3. ``zs``: input data
4. ``XI``: output x-coordinates for data grid
5. ``YI``: output y-coordinates for data grid
6. ``XR``: x-component of Barnes smoothing length scale
7. ``YR``: y-component of Barnes smoothing length scale

Keyword arguments
#################

- ``runs``: number of iterations

Returns
#######

- ``ZI``: interpolated data grid

References
##########

.. [Barnes1994a] S. L. Barnes, "Applications of the Barnes objective analysis scheme.  Part I:  Effects of undersampling, wave position, and station randomness," *Journal of Atmospheric and Oceanic Technology*, 11(6), 1433--1448, (1994).

.. [Barnes1994b] S. L. Barnes, "Applications of the Barnes objective analysis scheme.  Part II:  Improving derivative estimates," *Journal of Atmospheric and Oceanic Technology*, 11(6), 1449--1458, (1994).

.. [Barnes1994c] S. L. Barnes, "Applications of the Barnes objective analysis scheme.  Part III:  Tuning for minimum error," *Journal of Atmospheric and Oceanic Technology*, 11(6), 1459--1479, (1994).

.. [Daley1991] R. Daley, *Atmospheric data analysis*, Cambridge Press, New York.  (1991).
