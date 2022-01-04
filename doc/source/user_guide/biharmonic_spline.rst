====================
biharmonic_spline.py
====================

- Interpolates data using 2-dimensional biharmonic splines [Sandwell1987]_
- Can use surface splines in tension [Wessel1998]_
- Can use regularized surface splines [Mitasova1993]_

Calling Sequence
################

.. code-block:: python

    import spatial_interpolators as spi
    ZI = spi.biharmonic_spline(xs, ys, zs, XI, YI, tension=0.5)

`Source code`__

.. __: https://github.com/tsutterley/spatial-interpolators/blob/master/spatial_interpolators/biharmonic_spline.py


Arguments
#########

1. ``xs``: input x-coordinates
2. ``ys``: input y-coordinates
3. ``zs``: input data
4. ``XI``: output x-coordinates for data grid
5. ``YI``: output y-coordinates for data grid

Keyword arguments
#################

- ``metric``: distance metric to use (default euclidean)
- ``tension``: tension to use in interpolation
- ``regular``: use regularized function of [Mitasova1993]_
- ``eps``: minimum distance value for valid points
- ``scale``: scale factor for normalized lengths

Returns
#######

- ``ZI``: interpolated data grid

References
##########

.. [Sandwell1987] D. T. Sandwell, "Biharmonic spline interpolation of GEOS‐3 and SEASAT altimeter data," *Geophysical Research Letters*, 14(2), 139--142, (1987). `doi: 10.1029/GL014i002p00139 <https://doi.org/10.1029/GL014i002p00139>`_

.. [Mitasova1993] H. Mit\ |aacute|\ |scaron|\ ov\ |aacute| and L. Mit\ |aacute|\ |scaron|\ , "Interpolation by regularized spline with tension: I. Theory and implementation," *Mathematical Geology*, 25(6), 641--655, (1993). `doi: 10.1007/BF00893171 <https://doi.org/10.1007/BF00893171>`_

.. [Wessel1998] P. Wessel and D. Bercovici, "Interpolation with Splines in Tension: A Green's Function Approach," *Mathematical Geology*, 30(1), 77--93, (1998). `doi: 10.1023/A:1021713421882 <https://doi.org/10.1023/A:1021713421882>`_

.. |aacute|    unicode:: U+00E1 .. LATIN SMALL LETTER A WITH ACUTE

.. |scaron|    unicode:: U+0161 .. LATIN SMALL LETTER S WITH CARON
