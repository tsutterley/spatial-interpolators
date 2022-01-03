======================
shepard_interpolant.py
======================

- Interpolates data by evaluating Shepard Interpolants based on inverse distances [Shepard1968]_

Calling Sequence
################

.. code-block:: python

    import spatial_interpolators as spi
    ZI = spi.shepard_interpolant(xs, ys, zs, XI, YI, power=2.0)

`Source code`__

.. __: https://github.com/tsutterley/spatial-interpolators/blob/master/spatial_interpolators/shepard_interpolant.py


Arguments
#########

1. ``xs``: input x-coordinates
2. ``ys``: input y-coordinates
3. ``zs``: input data
4. ``XI``: output x-coordinates for data grid
5. ``YI``: output y-coordinates for data grid

Keyword arguments
#################

- ``power``: power used in the inverse distance weighting
- ``eps``: minimum distance value for valid points
- ``modified``: use declustering modified Shepard's interpolants [Schnell2014]_
- ``D``: declustering distance for modified algorithm
- ``L``: maximum distance to be included in weights with modified algorithm

Returns
#######

- ``ZI``: interpolated data grid

References
##########

.. [Schnell2014] J. Schnell, C. D. Holmes, A. Jangam, and M. J. Prather, "Skill in forecasting extreme ozone pollution episodes with a global atmospheric chemistry model," *Atmospheric Physics and chemistry*, 14(15), 7721--7739, (2014). `doi: 10.5194/acp-14-7721-2014 <https://doi.org/10.5194/acp-14-7721-2014>`_

.. [Shepard1968] D. Shepard, "A two-dimensional interpolation function for irregularly spaced data," *ACM68: Proceedings of the 1968 23rd ACM National Conference*, 517--524, (1968). `doi: 10.1145/800186.810616 <https://doi.org/10.1145/800186.810616>`_
