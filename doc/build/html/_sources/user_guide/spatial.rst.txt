==========
spatial.py
==========

Utilities for operating on spatial data

`Source code`__

.. __: https://github.com/tsutterley/spatial-interpolators/blob/master/spatial_interpolators/spatial.py

General Methods
===============


.. method:: spatial_interpolators.spatial.data_type(x, y, t)

    Determines input data type based on variable dimensions

    Arguments: spatial and temporal coordinates

    Returns:

        string denoting input data type

          * ``'time series'``
          * ``'drift'``
          * ``'grid'``


.. method:: spatial_interpolators.spatial.convert_ellipsoid(phi1, h1, a1, f1, a2, f2, eps=1e-12, itmax=10)

    Convert latitudes and heights to a different ellipsoid using Newton-Raphson

    Arguments:

        ``phi1``: latitude of input ellipsoid in degrees

        ``h1``: height above input ellipsoid in meters

        ``a1``: semi-major axis of input ellipsoid

        ``f1``: flattening of input ellipsoid

        ``a2``: semi-major axis of output ellipsoid

        ``f2``: flattening of output ellipsoid

    Keyword arguments:

        ``eps``: tolerance to prevent division by small numbers and to determine convergence

        ``itmax``: maximum number of iterations to use in Newton-Raphson

    Returns:

        ``phi2``: latitude of output ellipsoid in degrees

        ``h2``: height above output ellipsoid in meters


.. method:: spatial_interpolators.spatial.compute_delta_h(a1, f1, a2, f2, lat)

    Compute difference in elevation for two ellipsoids at a given latitude using a simplified empirical equation

    Arguments:

        ``a1``: semi-major axis of input ellipsoid

        ``f1``: flattening of input ellipsoid

        ``a2``: semi-major axis of output ellipsoid

        ``f2``: flattening of output ellipsoid

        ``lat``: array of latitudes in degrees

    Returns:

        ``delta_h``: difference in elevation for two ellipsoids


.. method:: spatial_interpolators.spatial.wrap_longitudes(lon):

    Wraps longitudes to range from -180 to +180

    Arguments:

        ``lon``: longitude


.. method:: spatial_interpolators.spatial.to_cartesian(lon,lat,a_axis=6378137.0,flat=1.0/298.257223563)

    Converts geodetic coordinates to Cartesian coordinates

    Arguments:

        ``lon``: longitude

        ``lat``: latitude

    Keyword arguments:

        ``h``: height

        ``a_axis``: semimajor axis of the ellipsoid

        ``flat``: ellipsoidal flattening

    Returns:

        ``x``, ``y``, ``z`` in Cartesian coordinates


.. method:: spatial_interpolators.spatial.to_sphere(x,y,z)

    Convert from Cartesian coordinates to spherical coordinates

    Arguments:

        ``x``, ``y``, ``z`` in Cartesian coordinates

    Returns:

        ``lon``: longitude

        ``lat``: latitude

        ``rad``: radius


.. method:: spatial_interpolators.spatial.to_geodetic(x,y,z,a_axis=6378137.0,flat=1.0/298.257223563)

    Convert from Cartesian coordinates to geodetic coordinates using `a closed form solution <https://arc.aiaa.org/doi/abs/10.2514/3.21016>`_

    Arguments:

        ``x``, ``y``, ``z`` in Cartesian coordinates

    Keyword arguments:

        ``a_axis``: semimajor axis of the ellipsoid

        ``flat``: ellipsoidal flattening

    Returns:

        ``lon``: longitude

        ``lat``: latitude

        ``h``: height


.. method:: spatial_interpolators.spatial.scale_areas(lat, flat=1.0/298.257223563, ref=70.0)

    Calculates area scaling factors for a polar stereographic projection

    Arguments:

        ``lat``: latitude

    Keyword arguments:

        ``flat``: ellipsoidal flattening

        ``ref``: reference latitude (true scale latitude)

    Returns:

        ``scale``: area scaling factors at input latitudes
