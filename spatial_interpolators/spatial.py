#!/usr/bin/env python
u"""
spatial.py
Written by Tyler Sutterley (04/2023)

Utilities for operating on spatial data

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html

UPDATE HISTORY:
    Updated 04/2023: copy inputs in cartesian to not modify original arrays
        added iterative methods for converting from cartesian to geodetic
    Updated 03/2023: add basic variable typing to function inputs
    Updated 04/2022: docstrings in numpy documentation format
    Updated 01/2022: use iteration breaks in convert ellipsoid function
    Updated 10/2021: add pole case in stereographic area scale calculation
    Updated 09/2021: can calculate height differences between ellipsoids
    Updated 07/2021: added function for determining input variable type
    Updated 03/2021: added polar stereographic area scale calculation
        add routines for converting to and from cartesian coordinates
        replaced numpy bool/int to prevent deprecation warnings
    Updated 12/2020: added module for converting ellipsoids
    Written 11/2020
"""
from __future__ import annotations

import numpy as np

def data_type(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> str:
    """
    Determines input data type based on variable dimensions

    Parameters
    ----------
    x: np.ndarray
        x-dimension coordinates
    y: np.ndarray
        y-dimension coordinates
    t: np.ndarray
        time-dimension coordinates

    Returns
    -------
    string denoting input data type

        - ``'time series'``
        - ``'drift'``
        - ``'grid'``
    """
    xsize = np.size(x)
    ysize = np.size(y)
    tsize = np.size(t)
    if (xsize == 1) and (ysize == 1) and (tsize >= 1):
        return 'time series'
    elif (xsize == ysize) & (xsize == tsize):
        return 'drift'
    elif (np.ndim(x) > 1) & (xsize == ysize):
        return 'grid'
    elif (xsize != ysize):
        return 'grid'
    else:
        raise ValueError('Unknown data type')

def convert_ellipsoid(
        phi1: np.ndarray,
        h1: np.ndarray,
        a1: float,
        f1: float,
        a2: float,
        f2: float,
        eps: float = 1e-12,
        itmax: int = 10
    ):
    """
    Convert latitudes and heights to a different ellipsoid using Newton-Raphson

    Parameters
    ----------
    phi1: np.ndarray
        latitude of input ellipsoid in degrees
    h1: np.ndarray
        height above input ellipsoid in meters
    a1: float
        semi-major axis of input ellipsoid
    f1: float
        flattening of input ellipsoid
    a2: float
        semi-major axis of output ellipsoid
    f2: float
        flattening of output ellipsoid
    eps: float, default 1e-12
        tolerance to prevent division by small numbers and
        to determine convergence
    itmax: int, default 10
        maximum number of iterations to use in Newton-Raphson

    Returns
    -------
    phi2: np.ndarray
        latitude of output ellipsoid in degrees
    h2: np.ndarray
        height above output ellipsoid in meters

    References
    ----------
    .. [1] J. Meeus, *Astronomical Algorithms*, 2nd edition, 477 pp., (1998).
    """
    if (len(phi1) != len(h1)):
        raise ValueError('phi and h have incompatable dimensions')
    # semiminor axis of input and output ellipsoid
    b1 = (1.0 - f1)*a1
    b2 = (1.0 - f2)*a2
    # initialize output arrays
    npts = len(phi1)
    phi2 = np.zeros((npts))
    h2 = np.zeros((npts))
    # for each point
    for N in range(npts):
        # force phi1 into range -90 <= phi1 <= 90
        if (np.abs(phi1[N]) > 90.0):
            phi1[N] = np.sign(phi1[N])*90.0
        # handle special case near the equator
        # phi2 = phi1 (latitudes congruent)
        # h2 = h1 + a1 - a2
        if (np.abs(phi1[N]) < eps):
            phi2[N] = np.copy(phi1[N])
            h2[N] = h1[N] + a1 - a2
        # handle special case near the poles
        # phi2 = phi1 (latitudes congruent)
        # h2 = h1 + b1 - b2
        elif ((90.0 - np.abs(phi1[N])) < eps):
            phi2[N] = np.copy(phi1[N])
            h2[N] = h1[N] + b1 - b2
        # handle case if latitude is within 45 degrees of equator
        elif (np.abs(phi1[N]) <= 45):
            # convert phi1 to radians
            phi1r = phi1[N] * np.pi/180.0
            sinphi1 = np.sin(phi1r)
            cosphi1 = np.cos(phi1r)
            # prevent division by very small numbers
            cosphi1 = np.copy(eps) if (cosphi1 < eps) else cosphi1
            # calculate tangent
            tanphi1 = sinphi1 / cosphi1
            u1 = np.arctan(b1 / a1 * tanphi1)
            hpr1sin = b1 * np.sin(u1) + h1[N] * sinphi1
            hpr1cos = a1 * np.cos(u1) + h1[N] * cosphi1
            # set initial value for u2
            u2 = np.copy(u1)
            # setup constants
            k0 = b2 * b2 - a2 * a2
            k1 = a2 * hpr1cos
            k2 = b2 * hpr1sin
            # perform newton-raphson iteration to solve for u2
            # cos(u2) will not be close to zero since abs(phi1) <= 45
            for i in range(0, itmax+1):
                cosu2 = np.cos(u2)
                fu2 = k0 * np.sin(u2) + k1 * np.tan(u2) - k2
                fu2p = k0 * cosu2 + k1 / (cosu2 * cosu2)
                if (np.abs(fu2p) < eps):
                    break
                else:
                    delta = fu2 / fu2p
                    u2 -= delta
                    if (np.abs(delta) < eps):
                        break
            # convert latitude to degrees and verify values between +/- 90
            phi2r = np.arctan(a2 / b2 * np.tan(u2))
            phi2[N] = phi2r*180.0/np.pi
            if (np.abs(phi2[N]) > 90.0):
                phi2[N] = np.sign(phi2[N])*90.0
            # calculate height
            h2[N] = (hpr1cos - a2 * np.cos(u2)) / np.cos(phi2r)
        # handle final case where latitudes are between 45 degrees and pole
        else:
            # convert phi1 to radians
            phi1r = phi1[N] * np.pi/180.0
            sinphi1 = np.sin(phi1r)
            cosphi1 = np.cos(phi1r)
            # prevent division by very small numbers
            cosphi1 = np.copy(eps) if (cosphi1 < eps) else cosphi1
            # calculate tangent
            tanphi1 = sinphi1 / cosphi1
            u1 = np.arctan(b1 / a1 * tanphi1)
            hpr1sin = b1 * np.sin(u1) + h1[N] * sinphi1
            hpr1cos = a1 * np.cos(u1) + h1[N] * cosphi1
            # set initial value for u2
            u2 = np.copy(u1)
            # setup constants
            k0 = a2 * a2 - b2 * b2
            k1 = b2 * hpr1sin
            k2 = a2 * hpr1cos
            # perform newton-raphson iteration to solve for u2
            # sin(u2) will not be close to zero since abs(phi1) > 45
            for i in range(0, itmax+1):
                sinu2 = np.sin(u2)
                fu2 = k0 * np.cos(u2) + k1 / np.tan(u2) - k2
                fu2p = -1 * (k0 * sinu2 + k1 / (sinu2 * sinu2))
                if (np.abs(fu2p) < eps):
                    break
                else:
                    delta = fu2 / fu2p
                    u2 -= delta
                    if (np.abs(delta) < eps):
                        break
            # convert latitude to degrees and verify values between +/- 90
            phi2r = np.arctan(a2 / b2 * np.tan(u2))
            phi2[N] = phi2r*180.0/np.pi
            if (np.abs(phi2[N]) > 90.0):
                phi2[N] = np.sign(phi2[N])*90.0
            # calculate height
            h2[N] = (hpr1sin - b2 * np.sin(u2)) / np.sin(phi2r)

    # return the latitude and height
    return (phi2, h2)

def compute_delta_h(
        a1: float,
        f1: float,
        a2: float,
        f2: float,
        lat: np.ndarray
    ):
    """
    Compute difference in elevation for two ellipsoids at a given
        latitude using a simplified empirical equation

    Parameters
    ----------
    a1: float
        semi-major axis of input ellipsoid
    f1: float
        flattening of input ellipsoid
    a2: float
        semi-major axis of output ellipsoid
    f2: float
        flattening of output ellipsoid
    lat: np.ndarray
        latitudes (degrees north)

    Returns
    -------
    delta_h: np.ndarray
        difference in elevation for two ellipsoids

    References
    ----------
    .. [1] J Meeus, *Astronomical Algorithms*, pp. 77--82, (1991).
    """
    # force phi into range -90 <= phi <= 90
    gt90, = np.nonzero((lat < -90.0) | (lat > 90.0))
    lat[gt90] = np.sign(lat[gt90])*90.0
    # semiminor axis of input and output ellipsoid
    b1 = (1.0 - f1)*a1
    b2 = (1.0 - f2)*a2
    # compute delta_a and delta_b coefficients
    delta_a = a2 - a1
    delta_b = b2 - b1
    # compute differences between ellipsoids
    # delta_h = -(delta_a * cos(phi)^2 + delta_b * sin(phi)^2)
    phi = lat * np.pi/180.0
    delta_h = -(delta_a*np.cos(phi)**2 + delta_b*np.sin(phi)**2)
    return delta_h

def wrap_longitudes(lon: float | np.ndarray):
    """
    Wraps longitudes to range from -180 to +180

    Parameters
    ----------
    lon: float or np.ndarray
        longitude (degrees east)
    """
    phi = np.arctan2(np.sin(lon*np.pi/180.0), np.cos(lon*np.pi/180.0))
    # convert phi from radians to degrees
    return phi*180.0/np.pi

def to_cartesian(
        lon: np.ndarray,
        lat: np.ndarray,
        h: float | np.ndarray = 0.0,
        a_axis: float = 6378137.0,
        flat: float = 1.0/298.257223563
    ):
    """
    Converts geodetic coordinates to Cartesian coordinates

    Parameters
    ----------
    lon: np.ndarray
        longitude (degrees east)
    lat: np.ndarray
        latitude (degrees north)
    h: float or np.ndarray, default 0.0
        height above ellipsoid (or sphere)
    a_axis: float, default 6378137.0
        semimajor axis of the ellipsoid

        for spherical coordinates set to radius of the Earth
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening

        for spherical coordinates set to 0
    """
    # verify axes and copy to not modify inputs
    lon = np.atleast_1d(np.copy(lon))
    lat = np.atleast_1d(np.copy(lat))
    # fix coordinates to be 0:360
    lon[lon < 0] += 360.0
    # Linear eccentricity and first numerical eccentricity
    lin_ecc = np.sqrt((2.0*flat - flat**2)*a_axis**2)
    ecc1 = lin_ecc/a_axis
    # convert from geodetic latitude to geocentric latitude
    dtr = np.pi/180.0
    # geodetic latitude in radians
    latitude_geodetic_rad = lat*dtr
    # prime vertical radius of curvature
    N = a_axis/np.sqrt(1.0 - ecc1**2.0*np.sin(latitude_geodetic_rad)**2.0)
    # calculate X, Y and Z from geodetic latitude and longitude
    X = (N + h) * np.cos(latitude_geodetic_rad) * np.cos(lon*dtr)
    Y = (N + h) * np.cos(latitude_geodetic_rad) * np.sin(lon*dtr)
    Z = (N * (1.0 - ecc1**2.0) + h) * np.sin(latitude_geodetic_rad)
    # return the cartesian coordinates
    return (X, Y, Z)

def to_sphere(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """
    Convert from cartesian coordinates to spherical coordinates

    Parameters
    ----------
    x, np.ndarray
        cartesian x-coordinates
    y, np.ndarray
        cartesian y-coordinates
    z, np.ndarray
        cartesian z-coordinates
    """
    # verify axes and copy to not modify inputs
    x = np.atleast_1d(np.copy(x))
    y = np.atleast_1d(np.copy(y))
    z = np.atleast_1d(np.copy(z))
    # calculate radius
    rad = np.sqrt(x**2.0 + y**2.0 + z**2.0)
    # calculate angular coordinates
    # phi: azimuthal angle
    phi = np.arctan2(y, x)
    # th: polar angle
    th = np.arccos(z/rad)
    # convert to degrees and fix to 0:360
    lon = 180.0*phi/np.pi
    if np.any(lon < 0):
        lt0 = np.nonzero(lon < 0)
        lon[lt0] += 360.0
    # convert to degrees and fix to -90:90
    lat = 90.0 - (180.0*th/np.pi)
    np.clip(lat, -90, 90, out=lat)
    # return latitude, longitude and radius
    return (lon, lat, rad)

def to_geodetic(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        a_axis: float = 6378137.0,
        flat: float = 1.0/298.257223563,
        method: str = 'bowring',
        eps: float = np.finfo(np.float64).eps,
        iterations: int = 10
    ):
    """
    Convert from cartesian coordinates to geodetic coordinates
    using either iterative or closed-form methods

    Parameters
    ----------
    x, float
        cartesian x-coordinates
    y, float
        cartesian y-coordinates
    z, float
        cartesian z-coordinates
    a_axis: float, default 6378137.0
        semimajor axis of the ellipsoid
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening
    method: str, default 'bowring'
        method to use for conversion

            - ``'moritz'``: iterative solution
            - ``'bowring'``: iterative solution
            - ``'zhu'``: closed-form solution
    eps: float, default np.finfo(np.float64).eps
        tolerance for iterative methods
    iterations: int, default 10
        maximum number of iterations
    """
    # verify axes and copy to not modify inputs
    x = np.atleast_1d(np.copy(x))
    y = np.atleast_1d(np.copy(y))
    z = np.atleast_1d(np.copy(z))
    # calculate the geodetic coordinates using the specified method
    if (method.lower() == 'moritz'):
        return _moritz_iterative(x, y, z, a_axis=a_axis, flat=flat,
            eps=eps, iterations=iterations)
    elif (method.lower() == 'bowring'):
        return _bowring_iterative(x, y, z, a_axis=a_axis, flat=flat,
            eps=eps, iterations=iterations)
    elif (method.lower() == 'zhu'):
        return _zhu_closed_form(x, y, z, a_axis=a_axis, flat=flat)
    else:
        raise ValueError(f'Unknown conversion method: {method}')

def _moritz_iterative(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        a_axis: float = 6378137.0,
        flat: float = 1.0/298.257223563,
        eps: float = np.finfo(np.float64).eps,
        iterations: int = 10
    ):
    """
    Convert from cartesian coordinates to geodetic coordinates
    using the iterative solution of [1]_

    Parameters
    ----------
    x, float
        cartesian x-coordinates
    y, float
        cartesian y-coordinates
    z, float
        cartesian z-coordinates
    a_axis: float, default 6378137.0
        semimajor axis of the ellipsoid
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening
    eps: float, default np.finfo(np.float64).eps
        tolerance for iterative method
    iterations: int, default 10
        maximum number of iterations

    References
    ----------
    .. [1] B. Hofmann-Wellenhof and H. Moritz,
        *Physical Geodesy*, 2nd Edition, 403 pp., (2006).
        `doi: 10.1007/978-3-211-33545-1
        <https://doi.org/10.1007/978-3-211-33545-1>`_
    """
    # Linear eccentricity and first numerical eccentricity
    lin_ecc = np.sqrt((2.0*flat - flat**2)*a_axis**2)
    ecc1 = lin_ecc/a_axis
    # degrees to radians
    dtr = np.pi/180.0
    # calculate longitude
    lon = np.arctan2(y, x)/dtr
    # set initial estimate of height to 0
    h = np.zeros_like(lon)
    h0 = np.inf*np.ones_like(lon)
    # calculate radius of parallel
    p = np.sqrt(x**2 + y**2)
    # initial estimated value for phi using h=0
    phi = np.arctan(z/(p*(1.0 - ecc1**2)))
    # iterate to tolerance or to maximum number of iterations
    i = 0
    while np.any(np.abs(h - h0) > eps) and (i <= iterations):
        # copy previous iteration of height
        h0 = np.copy(h)
        # calculate radius of curvature
        N = a_axis/np.sqrt(1.0 - ecc1**2 * np.sin(phi)**2)
        # estimate new value of height
        h = p/np.cos(phi) - N
        # estimate new value for latitude using heights
        phi = np.arctan(z/(p*(1.0 - ecc1**2*N/(N + h))))
        # add to iterator
        i += 1
    # return latitude, longitude and height
    return (lon, phi/dtr, h)

def _bowring_iterative(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        a_axis: float = 6378137.0,
        flat: float = 1.0/298.257223563,
        eps: float = np.finfo(np.float64).eps,
        iterations: int = 10
    ):
    """
    Convert from cartesian coordinates to geodetic coordinates
    using the iterative solution of [1]_ [2]_

    Parameters
    ----------
    x, float
        cartesian x-coordinates
    y, float
        cartesian y-coordinates
    z, float
        cartesian z-coordinates
    a_axis: float, default 6378137.0
        semimajor axis of the ellipsoid
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening
    eps: float, default np.finfo(np.float64).eps
        tolerance for iterative method
    iterations: int, default 10
        maximum number of iterations

    References
    ----------
    .. [1] B. R. Bowring, "Transformation from spatial
        to geodetic coordinates," *Survey Review*, 23(181),
        323--327, (1976). `doi: 10.1179/sre.1976.23.181.323
        <https://doi.org/10.1179/sre.1976.23.181.323>`_
    .. [2] B. R. Bowring, "The Accuracy Of Geodetic
        Latitude and Height Equations," *Survey Review*, 28(218),
        202--206, (1985). `doi: 10.1179/sre.1985.28.218.202
        <https://doi.org/10.1179/sre.1985.28.218.202>`_
    """
    # semiminor axis of the WGS84 ellipsoid [m]
    b_axis = (1.0 - flat)*a_axis
    # Linear eccentricity
    lin_ecc = np.sqrt((2.0*flat - flat**2)*a_axis**2)
    # square of first and second numerical eccentricity
    e12 = lin_ecc**2/a_axis**2
    e22 = lin_ecc**2/b_axis**2
    # degrees to radians
    dtr = np.pi/180.0
    # calculate longitude
    lon = np.arctan2(y, x)/dtr
    # calculate radius of parallel
    p = np.sqrt(x**2 + y**2)
    # initial estimated value for reduced parametric latitude
    u = np.arctan(a_axis*z/(b_axis*p))
    # initial estimated value for latitude
    phi = np.arctan((z + e22*b_axis*np.sin(u)**3) /
        (p - e12*a_axis*np.cos(u)**3))
    phi0 = np.inf*np.ones_like(lon)
    # iterate to tolerance or to maximum number of iterations
    i = 0
    while np.any(np.abs(phi - phi0) > eps) and (i <= iterations):
        # copy previous iteration of phi
        phi0 = np.copy(phi)
        # calculate reduced parametric latitude
        u = np.arctan(b_axis*np.tan(phi)/a_axis)
        # estimate new value of latitude
        phi = np.arctan((z + e22*b_axis*np.sin(u)**3) /
            (p - e12*a_axis*np.cos(u)**3))
        # add to iterator
        i += 1
    # calculate final radius of curvature
    N = a_axis/np.sqrt(1.0 - e12 * np.sin(phi)**2)
    # estimate final height (Bowring, 1985)
    h = p*np.cos(phi) + z*np.sin(phi) - a_axis**2/N
    # return latitude, longitude and height
    return (lon, phi/dtr, h)

def _zhu_closed_form(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        a_axis: float = 6378137.0,
        flat: float = 1.0/298.257223563,
    ):
    """
    Convert from cartesian coordinates to geodetic coordinates
    using the closed-form solution of [1]_

    Parameters
    ----------
    x, float
        cartesian x-coordinates
    y, float
        cartesian y-coordinates
    z, float
        cartesian z-coordinates
    a_axis: float, default 6378137.0
        semimajor axis of the ellipsoid
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening

    References
    ----------
    .. [1] J. Zhu, "Exact conversion of Earth-centered,
        Earth-fixed coordinates to geodetic coordinates,"
        *Journal of Guidance, Control, and Dynamics*,
        16(2), 389--391, (1993). `doi: 10.2514/3.21016
        <https://arc.aiaa.org/doi/abs/10.2514/3.21016>`_
    """
    # semiminor axis of the WGS84 ellipsoid [m]
    b_axis = (1.0 - flat)*a_axis
    # Linear eccentricity
    lin_ecc = np.sqrt((2.0*flat - flat**2)*a_axis**2)
    # square of first numerical eccentricity
    e12 = lin_ecc**2/a_axis**2
    # degrees to radians
    dtr = np.pi/180.0
    # calculate longitude
    lon = np.arctan2(y, x)/dtr
    # calculate radius of parallel
    w = np.sqrt(x**2 + y**2)
    # allocate for output latitude and height
    lat = np.zeros_like(lon)
    h = np.zeros_like(lon)
    if np.any(w == 0):
        # special case where w == 0 (exact polar solution)
        ind, = np.nonzero(w == 0)
        h[ind] = np.sign(z[ind])*z[ind] - b_axis
        lat[ind] = 90.0*np.sign(z[ind])
    else:
        # all other cases
        ind, = np.nonzero(w != 0)
        l = e12/2.0
        m = (w[ind]/a_axis)**2.0
        n = ((1.0 - e12)*z[ind]/b_axis)**2.0
        i = -(2.0*l**2 + m + n)/2.0
        k = (l**2.0 - m - n)*l**2.0
        q = (1.0/216.0)*(m + n - 4.0*l**2)**3.0 + m*n*l**2.0
        D = np.sqrt((2.0*q - m*n*l**2)*m*n*l**2)
        B = i/3.0 - (q + D)**(1.0/3.0) - (q - D)**(1.0/3.0)
        t = np.sqrt(np.sqrt(B**2-k) - (B + i)/2.0) - \
            np.sign(m - n)*np.sqrt((B - i)/2.0)
        wi = w/(t + l)
        zi = (1.0 - e12)*z[ind]/(t - l)
        # calculate latitude and height
        lat[ind] = np.arctan2(zi, ((1.0 - e12)*wi))/dtr
        h[ind] = np.sign(t-1.0+l)*np.sqrt((w-wi)**2.0 + (z[ind]-zi)**2.0)
    # return latitude, longitude and height
    return (lon, lat, h)

def scale_areas(
        lat: np.ndarray,
        flat: float = 1.0/298.257223563,
        ref: float = 70.0
    ):
    """
    Calculates area scaling factors for a polar stereographic projection
    including special case of at the exact pole [1]_ [2]_

    Parameters
    ----------
    lat: np.ndarray
        latitude (degrees north)
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening
    ref: float, default 70.0
        reference latitude (true scale latitude)

    Returns
    -------
    scale: np.ndarray
        area scaling factors at input latitudes

    References
    ----------
    .. [1] J. P. Snyder, *Map Projections used by the U.S. Geological Survey*,
        Geological Survey Bulletin 1532, U.S. Government Printing Office, (1982).
    .. [2] JPL Technical Memorandum 3349-85-101
    """
    # convert latitude from degrees to positive radians
    theta = np.abs(lat)*np.pi/180.0
    # convert reference latitude from degrees to positive radians
    theta_ref = np.abs(ref)*np.pi/180.0
    # square of the eccentricity of the ellipsoid
    # ecc2 = (1-b**2/a**2) = 2.0*flat - flat^2
    ecc2 = 2.0*flat - flat**2
    # eccentricity of the ellipsoid
    ecc = np.sqrt(ecc2)
    # calculate ratio at input latitudes
    m = np.cos(theta)/np.sqrt(1.0 - ecc2*np.sin(theta)**2)
    t = np.tan(np.pi/4.0 - theta/2.0)/((1.0 - ecc*np.sin(theta)) / \
        (1.0 + ecc*np.sin(theta)))**(ecc/2.0)
    # calculate ratio at reference latitude
    mref = np.cos(theta_ref)/np.sqrt(1.0 - ecc2*np.sin(theta_ref)**2)
    tref = np.tan(np.pi/4.0 - theta_ref/2.0)/((1.0 - ecc*np.sin(theta_ref)) / \
        (1.0 + ecc*np.sin(theta_ref)))**(ecc/2.0)
    # distance scaling
    k = (mref/m)*(t/tref)
    kp = 0.5*mref*np.sqrt(((1.0+ecc)**(1.0+ecc))*((1.0-ecc)**(1.0-ecc)))/tref
    # area scaling
    scale = np.where(np.isclose(theta, np.pi/2.0), 1.0/(kp**2), 1.0/(k**2))
    return scale
