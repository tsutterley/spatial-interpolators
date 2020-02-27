#!/usr/bin/env python
u"""
shepard_interpolant.py
Written by Tyler Sutterley (09/2016)

Evaluates Shepard interpolants to 2D data based on inverse distance weighting
Resultant output will not be as accurate as radial basis functions but is faster

CALLING SEQUENCE:
    ZI = shepard_interpolant(xs, ys, zs, XI, YI)
    ZI = shepard_interpolant(xs, ys, zs, XI, YI, MODIFIED=True, D=25e3, L=500e3)

INPUTS:
    xs: input X data
    ys: input Y data
    zs: input data (Z variable)
    XI: grid X for output ZI (or array)
    YI: grid Y for output ZI (or array)
OUTPUTS:
    ZI: interpolated grid (or array)
OPTIONS:
    POWER: Power used in the inverse distance weighting (positive real number)
    EPS: minimum distance value for valid points (default 1e-7)
    MODIFIED: use declustering modified Shepard's interpolants
    D: declustering distance
    L: maximum distance to be included in weights

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python (https://numpy.org)

REFERENCES:
    Donald Shepard, A two-dimensional interpolation function for irregularly
        spaced data, ACM68: Proceedings of the 1968 23rd ACM National Conference
    Schnell et al., Skill in forecasting extreme ozone pollution episodes with
        a global atmospheric chemistry model. Atmos. Chem. Phys., 14, 7721-7739,
        doi:10.5194/acp-14-7721-2014, 2014

UPDATE HISTORY:
    Updated 09/2016: added modified Shepard's interpolants for declustering
        following Schnell et al (2014)
    Written 08/2016
"""
import numpy as np

def shepard_interpolant(xs, ys, zs, XI, YI, POWER=0.0, EPS=1e-7,
    MODIFIED=False, D=25e3, L=500e3):
    #-- remove singleton dimensions
    xs = np.squeeze(xs)
    ys = np.squeeze(ys)
    zs = np.squeeze(zs)
    XI = np.squeeze(XI)
    YI = np.squeeze(YI)
    #-- number of data points
    npts = len(zs)
    #-- size of new matrix
    if (np.ndim(XI) == 1):
        ni = len(XI)
    else:
        nx,ny = np.shape(XI)
        ni = XI.size

    #-- Check to make sure sizes of input arguments are correct and consistent
    if (len(zs) != len(xs)) | (len(zs) != len(ys)):
        raise Exception('Length of X, Y, and Z must be equal')
    if (np.shape(XI) != np.shape(YI)):
        raise Exception('Size of XI and YI must be equal')
    if (POWER < 0):
        raise ValueError('Power parameter must be positive')

    #-- Modified Shepard interpolants for declustering data
    if MODIFIED:
        #-- calculate number of points within distance D of each point
        M = np.zeros((npts))
        for i,XY in enumerate(zip(xs,ys)):
            #-- compute radial distance between data point and data coordinates
            Rd = np.sqrt((XY[0] - xs)**2 + (XY[1] - ys)**2)
            M[i] = np.count_nonzero(Rd <= D)

    #-- for each interpolated value
    ZI = np.zeros((ni))
    for i,XY in enumerate(zip(XI.flatten(),YI.flatten())):
        #-- compute the radial distance between point i and data coordinates
        Re = np.sqrt((XY[0] - xs)**2 + (XY[1] - ys)**2)
        #-- Modified Shepard interpolants for declustering data
        if MODIFIED:
            #-- calculate weights
            w = np.zeros((npts))
            #-- find indices of cases
            ind_D = np.nonzero(Re < D)
            ind_M = np.nonzero((Re >= D) & (Re < L))
            ind_L = np.nonzero((Re >= L))
            #-- declustering of close points (weighted equally)
            w[ind_D] = D**(-POWER)/M[ind_D]
            #-- inverse distance weighting of mid-range points with scaling
            power_inverse_distance = Re[ind_M]**(-POWER)
            w[ind_M] = power_inverse_distance/M[ind_M]
            #-- no weight of distant points
            w[ind_L] = 0.0
            #-- calculate sum of all weights
            s = np.sum(w)
            #-- Find 2D interpolated surface
            ZI[i] = np.dot(w/s, zs) if (s > 0.0) else np.nan
        elif (Re < EPS).any():
            #-- if a data coordinate is within the EPS cutoff
            min_indice, = np.nonzero(Re < EPS)
            ZI[i] = zs[min_indice]
        else:
            #-- compute the weights based on POWER
            if (POWER == 0.0):
                #-- weights if POWER is 0
                w = np.ones((npts))/npts
            else:
                #-- normalized weights if POWER > 0 (typically between 1 and 3)
                #-- in the inverse distance weighting
                power_inverse_distance = Re**(-POWER)
                s = np.sum(power_inverse_distance)
                w = power_inverse_distance/s
            #-- Find 2D interpolated surface
            ZI[i] = np.dot(w, zs)

    #-- reshape to original dimensions
    if (np.ndim(XI) != 1):
        ZI = ZI.reshape(nx,ny)
    #-- return output matrix/array
    return ZI
