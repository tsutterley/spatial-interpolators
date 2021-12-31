#!/usr/bin/env python
u"""
biharmonic_spline.py
Written by Tyler Sutterley (07/2021)

Interpolates a sparse grid using 2D biharmonic splines (Sandwell, 1987)
With or without tension parameters (Wessel and Bercovici, 1998)
or using the regularized function of Mitasova and Mitas 1993

CALLING SEQUENCE:
    ZI = biharmonic_spline(xs, ys, zs, XI, YI)

INPUTS:
    xs: input X data
    ys: input Y data
    zs: input data (Z variable)
    XI: grid X for output ZI (or array)
    YI: grid Y for output ZI (or array)

OUTPUTS:
    ZI: interpolated grid (or array)

OPTIONS:
    METRIC: distance metric to use (default euclidean)
    TENSION: tension to use in interpolation (between 0 and 1)
    REGULAR: use regularized function of Mitasova and Mitas
    EPS: minimum distance value for valid points (default 1e-7)

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python (https://numpy.org)
    scipy: Scientific Tools for Python (https://docs.scipy.org/doc/)

REFERENCES:
    Sandwell, D. T. (1987), Biharmonic spline interpolation of GEOS-3 and
        SEASAT altimeter data, Geophysical Research Letters, Vol. 2.
    Wessel and Bercovici (1998), Interpolation with Splines in Tension: A
        Green's Function Approach, Mathematical Geology, Vol. 30, No. 1.
    Mitasova and Mitas (1993), Mathematical Geology, Vol. 25, No. 6

UPDATE HISTORY:
    Updated 07/2021: using scipy spatial distance routines
    Updated 09/2017: use rcond=-1 in numpy least-squares algorithms
    Updated 08/2016: detrend input data and retrend output data. calculate c
        added regularized function of Mitasova and Mitas
    Updated 06/2016: added TENSION parameter (Wessel and Bercovici, 1998)
    Written 06/2016
"""
import numpy as np
import scipy.spatial

def biharmonic_spline(xs, ys, zs, XI, YI, METRIC='euclidean',
    TENSION=0, REGULAR=False, EPS=1e-7):
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
    if (TENSION < 0) or (TENSION >= 1):
        raise ValueError('TENSION must be greater than 0 and less than 1')

    #-- Compute GG matrix for GG*m = d inversion problem
    npts = len(zs)
    GG = np.zeros((npts,npts))
    #-- Computation of distance Matrix (data to data)
    if (METRIC == 'brute'):
        #-- use linear algebra to compute euclidean distances
        Rd = distance_matrix(
            np.array([xs, ys]),
            np.array([xs, ys])
            )
    else:
        #-- use scipy spatial distance routines
        Rd = scipy.spatial.distance.cdist(
            np.array([xs, ys]).T,
            np.array([xs, ys]).T,
            metric=METRIC)
    #-- Calculate length scale for regularized case (Mitasova and Mitas)
    length_scale = np.sqrt((XI.max() - XI.min())**2 + (YI.max() - YI.min())**2)
    #-- calculate Green's function for valid points (with or without tension)
    ii,jj = np.nonzero(Rd >= EPS)
    if (TENSION == 0):
        GG[ii,jj] = (Rd[ii,jj]**2) * (np.log(Rd[ii,jj]) - 1.0)
    elif REGULAR:
        GG[ii,jj] = regular_spline2D(Rd[ii,jj], TENSION, length_scale/50.0)
    else:
        GG[ii,jj] = green_spline2D(Rd[ii,jj], TENSION)
    #-- detrend dataset
    z0,r0,p = detrend2D(xs,ys,zs)
    #-- Compute model m for detrended data
    m = np.linalg.lstsq(GG,z0,rcond=-1)[0]

    #-- Computation of distance Matrix (data to mesh points)
    if (METRIC == 'brute'):
        #-- use linear algebra to compute euclidean distances
        Re = distance_matrix(
            np.array([XI.flatten(),YI.flatten()]),
            np.array([xs,ys])
            )
    else:
        #-- use scipy spatial distance routines
        Re = scipy.spatial.distance.cdist(
            np.array([XI.flatten(),YI.flatten()]).T,
            np.array([xs, ys]).T,
            metric=METRIC)
    gg = np.zeros_like(Re)
    #-- calculate Green's function for valid points (with or without tension)
    ii,jj = np.nonzero(Re >= EPS)
    if (TENSION == 0):
        gg[ii,jj] = (Re[ii,jj]**2) * (np.log(Re[ii,jj]) - 1.0)
    elif REGULAR:
        gg[ii,jj] = regular_spline2D(Re[ii,jj], TENSION, length_scale/50.0)
    else:
        gg[ii,jj] = green_spline2D(Re[ii,jj], TENSION)

    #-- Find 2D interpolated surface through irregular/regular X, Y grid points
    if (np.ndim(XI) == 1):
        ZI = np.squeeze(np.dot(gg,m))
    else:
        ZI = np.zeros((nx,ny))
        ZI[:,:] = np.dot(gg,m).reshape(nx,ny)
    #-- return output matrix after retrending
    return (ZI + r0[2]) + (XI-r0[0])*p[0] + (YI-r0[1])*p[1]

#-- Removing mean and slope in 2-D dataset
#-- http://www.soest.hawaii.edu/wessel/tspline/
def detrend2D(xi, yi, zi):
    #-- Find mean values
    r0 = np.zeros((3))
    r0[0] = xi.mean()
    r0[1] = yi.mean()
    r0[2] = zi.mean()
    #-- Extract mean values from X, Y and Z
    x0 = xi - r0[0]
    y0 = yi - r0[1]
    z0 = zi - r0[2]
    #-- Find slope parameters
    p = np.linalg.lstsq(np.transpose([x0,y0]),z0,rcond=-1)[0]
    #-- Extract slope from data
    z0 = z0 - x0*p[0] - y0*p[1]
    #-- return the detrended value, the mean values, and the slope parameters
    return (z0, r0, p)

#-- calculate Euclidean distances between points as matrices
def distance_matrix(x,cntrs):
    s,M = np.shape(x)
    s,N = np.shape(cntrs)
    D = np.zeros((M,N))
    for d in range(s):
        ii, = np.dot(d,np.ones((1,N))).astype(np.int)
        jj, = np.dot(d,np.ones((1,M))).astype(np.int)
        dx = x[ii,:].transpose() - cntrs[jj,:]
        D += dx**2
    D = np.sqrt(D)
    return D

#-- Green function for 2-D spline in tension (Wessel et al, 1998)
#-- http://www.soest.hawaii.edu/wessel/tspline/
def green_spline2D(x, t):
    #-- in tension: G(u) = G(u) - log(u)
    #-- where u = c * x and c = sqrt (t/(1-t))
    c = np.sqrt(t/(1.0 - t))
    #-- allocate for output Green's function
    G = np.zeros_like(x)
    #-- inverse of tension parameter
    inv_c = 1.0/c
    #-- log(2) - 0.5772156
    g0 = 0.115931515658412420677337
    #-- find points below (or equal to) 2 times inverse tension parameter
    ii, = np.nonzero(x <= (2.0*inv_c))
    u = c*x[ii]
    y = 0.25*(u**2)
    z = (u**2)/14.0625
    #-- Green's function for points ii (less than or equal to 2.0*c)
    G[ii] = (-np.log(0.5*u) * (z * (3.5156229 + z * (3.0899424 + z * \
        (1.2067492 + z * (0.2659732 + z * (0.360768e-1 + z*0.45813e-2))))))) + \
        (y * (0.42278420 + y * (0.23069756 + y * (0.3488590e-1 + \
        y * (0.262698e-2 + y * (0.10750e-3 + y * 0.74e-5))))))
    #-- find points above 2 times inverse tension parameter
    ii, = np.nonzero(x > 2.0*inv_c)
    y = 2.0*inv_c/x[ii]
    u = c*x[ii]
    #-- Green's function for points ii (greater than 2.0*c)
    G[ii] = (np.exp(-u)/np.sqrt(u)) * (1.25331414 + y * (-0.7832358e-1 + y * \
        (0.2189568e-1 + y * (-0.1062446e-1 + y * (0.587872e-2 + y * \
        (-0.251540e-2 + y * 0.53208e-3)))))) + np.log(u) - g0
    return G

#-- Regularized spline in tension (Mitasova and Mitas, 1993)
def regular_spline2D(r, t, l):
    #-- calculate tension parameter
    p = np.sqrt(t/(1.0 - t))/l
    z = (0.5 * p * r)**2
    #-- allocate for output Green's function
    G = np.zeros_like(r)
    #-- Green's function for points A (less than or equal to 1)
    A = np.nonzero(z <= 1.0)
    G[A] =  0.99999193*z[A]
    G[A] -= 0.24991055*z[A]**2
    G[A] += 0.05519968*z[A]**3
    G[A] -= 0.00976004*z[A]**4
    G[A] += 0.00107857*z[A]**4
    #-- Green's function for points B (greater than 1)
    B = np.nonzero(z > 1.0)
    En = 0.2677737343 +  8.6347608925 * z[B]
    Ed = 3.9584869228 + 21.0996530827 * z[B]
    En += 18.0590169730 * z[B]**2
    Ed += 25.6329561486 * z[B]**2
    En += 8.5733287401 * z[B]**3
    Ed += 9.5733223454 * z[B]**3
    En += z[B]**4
    Ed += z[B]**4
    G[B] = np.log(z[B]) + 0.577215664901 + (En/Ed)/(z[B]*np.exp(z[B]))
    return G
