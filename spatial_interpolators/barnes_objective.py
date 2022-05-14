#!/usr/bin/env python
u"""
barnes_objective.py
Written by Tyler Sutterley (05/2022)

Barnes objective analysis for the optimal interpolation
of an input grid using a successive corrections scheme

CALLING SEQUENCE:
    ZI = barnes_objective(xs, ys, zs, XI, YI, XR, YR)
    ZI = barnes_objective(xs, ys, zs, XI, YI, XR, YR, RUNS=3)

INPUTS:
    xs: input X data
    ys: input Y data
    zs: input data
    XI: grid X for output ZI
    YI: grid Y for output ZI
    XR: x component of Barnes smoothing length scale
        Remains fixed throughout the iterations
    YR: y component of Barnes smoothing length scale
        Remains fixed throughout the iterations

OUTPUTS:
    ZI: interpolated grid

OPTIONS:
    runs: number of iterations

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org

REFERENCES:
    S. L. Barnes, Applications of the Barnes objective analysis scheme.
        Part I: effects of undersampling, wave position, and station
        randomness. J. of Atmos. and Oceanic Tech., 11, 1433-1448. (1994)
    S. L. Barnes, Applications of the Barnes objective analysis scheme.
        Part II: Improving derivative estimates. J. of Atmos. and
        Oceanic Tech., 11, 1449-1458. (1994_
    S. L. Barnes, Applications of the Barnes objective analysis scheme.
        Part III: Tuning for minimum error. J. of Atmos. and Oceanic
        Tech., 11, 1459-1479. (1994)
    R. Daley, Atmospheric data analysis, Cambridge Press, New York.
        Section 3.6. (1991)

UPDATE HISTORY:
    Updated 05/2022: updated docstrings to numpy documentation format
    Updated 01/2022: added function docstrings
    Written 08/2016
"""
import numpy as np

def barnes_objective(xs, ys, zs, XI, YI, XR, YR, runs=3):
    """
    Barnes objective analysis for the optimal interpolation
    of an input grid using a successive corrections scheme

    Parameters
    ----------
    xs: float
        input x-coordinates
    ys: float
        input y-coordinates
    zs: float
        input data
    XI: float
        output x-coordinates for data grid
    YI: float
        output y-coordinates for data grid
    XR: float
        x-component of Barnes smoothing length scale
    YR: float
        y-component of Barnes smoothing length scale
    runs: int, default 3
        number of iterations

    Returns
    -------
    ZI: float
        interpolated data grid

    References
    ----------
    .. [Barnes1994a] S. L. Barnes,
        "Applications of the Barnes objective analysis scheme.
        Part I:  Effects of undersampling, wave position, and
        station randomness," *Journal of Atmospheric and Oceanic
        Technology*, 11(6), 1433--1448, (1994).
    .. [Barnes1994b] S. L. Barnes,
        "Applications of the Barnes objective analysis scheme.
        Part II:  Improving derivative estimates,"
        *Journal of Atmospheric and Oceanic Technology*,
        11(6), 1449--1458, (1994).
    .. [Barnes1994c] S. L. Barnes,
        "Applications of the Barnes objective analysis scheme.
        Part III:  Tuning for minimum error,"
        *Journal of Atmospheric and Oceanic Technology*,
        11(6), 1459--1479, (1994).
    .. [Daley1991] R. Daley, *Atmospheric data analysis*,
        Cambridge Press, New York.  (1991).
    """
    #-- remove singleton dimensions
    xs = np.squeeze(xs)
    ys = np.squeeze(ys)
    zs = np.squeeze(zs)
    XI = np.squeeze(XI)
    YI = np.squeeze(YI)
    #-- size of new matrix
    if (np.ndim(XI) == 1):
        nx = len(XI)
    else:
        nx,ny = np.shape(XI)

    #-- Check to make sure sizes of input arguments are correct and consistent
    if (len(zs) != len(xs)) | (len(zs) != len(ys)):
        raise Exception('Length of X, Y, and Z must be equal')
    if (np.shape(XI) != np.shape(YI)):
        raise Exception('Size of XI and YI must be equal')

    #-- square of Barnes smoothing lengths scale
    xr2 = XR**2
    yr2 = YR**2
    #-- allocate for output zp array
    zp = np.zeros_like(XI.flatten())
    #-- first analysis
    for i,XY in enumerate(zip(XI.flatten(),YI.flatten())):
        dx = np.abs(xs - XY[0])
        dy = np.abs(ys - XY[1])
        #-- calculate weights
        w = np.exp(-dx**2/xr2 - dy**2/yr2)
        zp[i] = np.sum(zs*w)/sum(w)

    #-- allocate for even and odd zp arrays if iterating
    if (runs > 0):
        zpEven = np.zeros_like(zs)
        zpOdd = np.zeros_like(zs)
    #-- for each run
    for n in range(runs):
        #-- calculate even and odd zp arrays
        for j,xy in enumerate(zip(xs,ys)):
            dx = np.abs(xs - xy[0])
            dy = np.abs(ys - xy[1])
            #-- calculate weights
            w = np.exp(-dx**2/xr2 - dy**2/yr2)
            if ((n % 2) == 0):#-- even (% = modulus)
                zpEven[j] = zpOdd[j] + np.sum((zs - zpOdd)*w)/np.sum(w)
            else:#-- odd
                zpOdd[j] = zpEven[j] + np.sum((zs - zpEven)*w)/np.sum(w)
        #-- calculate zp for run n
        for i,XY in enumerate(zip(XI.flatten(),YI.flatten())):
            dx = np.abs(xs - XY[0])
            dy = np.abs(ys - XY[1])
            w = np.exp(-dx**2/xr2 - dy**2/yr2)
            if ((n % 2) == 0):#-- even (% = modulus)
                zp[i] = zp[i] + np.sum((zs - zpEven)*w)/np.sum(w)
            else:#-- odd
                zp[i] = zp[i] + np.sum((zs - zpOdd)*w)/np.sum(w)

    #-- reshape to original dimensions
    if (np.ndim(XI) != 1):
        ZI = zp.reshape(nx,ny)
    else:
        ZI = zp.copy()

    #-- return output matrix/array
    return ZI
