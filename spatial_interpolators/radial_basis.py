#!/usr/bin/env python
u"""
radial_basis.py
Written by Tyler Sutterley (07/2021)

Interpolates a sparse grid using radial basis functions

CALLING SEQUENCE:
    ZI = radial_basis(xs, ys, zs, XI, YI, polynomial=0,
        smooth=smooth, epsilon=epsilon, method='inverse')

INPUTS:
    xs: scaled input X data
    ys: scaled input Y data
    data: input data (Z variable)
    XI: scaled grid X for output ZI (or array)
    YI: scaled grid Y for output ZI (or array)

OUTPUTS:
    ZI: interpolated data grid (or array)

OPTIONS:
    smooth: smoothing weights
    metric: distance metric to use (default euclidean)
    epsilon: norm input
        default is mean Euclidean distance
    polynomial: polynomial order if augmenting radial basis functions
        default None: no polynomials
    method: radial basis function
        multiquadric
        inverse_multiquadric or inverse (default)
        inverse_quadratic
        gaussian
        linear (first-order polyharmonic spline)
        cubic (third-order polyharmonic spline)
        quintic (fifth-order polyharmonic spline)
        thin_plate: thin-plate spline

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python (https://numpy.org)
    scipy: Scientific Tools for Python (https://docs.scipy.org/doc/)

REFERENCES:
    R. L. Hardy, Multiquadric equations of topography and other irregular
        surfaces, J. Geophys. Res., 76(8), 1905-1915, 1971.
    M. Buhmann, "Radial Basis Functions", Cambridge Monographs on Applied and
        Computational Mathematics, 2003.

UPDATE HISTORY:
    Updated 07/2021: using scipy spatial distance routines
    Updated 09/2017: using rcond=-1 in numpy least-squares algorithms
    Updated 01/2017: epsilon in polyharmonic splines (linear, cubic, quintic)
    Updated 08/2016: using format text within ValueError, edit constant vector
        removed 3 dimensional option of radial basis (spherical)
        changed hierarchical_radial_basis to compact_radial_basis using
            compactly-supported radial basis functions and sparse matrices
        added low-order polynomial option (previously used default constant)
    Updated 01/2016: new hierarchical_radial_basis function
        that first reduces to points within distance.  added cutoff option
    Updated 10/2014: added third dimension (spherical)
    Written 08/2014
"""
from __future__ import print_function, division
import numpy as np
import scipy.spatial

def radial_basis(xs, ys, zs, XI, YI, smooth=0.0, metric='euclidean',
    epsilon=None, method='inverse', polynomial=None):
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

    #-- create python dictionary of radial basis function formulas
    radial_basis_functions = {}
    radial_basis_functions['multiquadric'] = multiquadric
    radial_basis_functions['inverse_multiquadric'] = inverse_multiquadric
    radial_basis_functions['inverse'] = inverse_multiquadric
    radial_basis_functions['inverse_quadratic'] = inverse_quadratic
    radial_basis_functions['gaussian'] = gaussian
    radial_basis_functions['linear'] = poly_spline1
    radial_basis_functions['cubic'] = poly_spline3
    radial_basis_functions['quintic'] = poly_spline5
    radial_basis_functions['thin_plate'] = thin_plate
    #-- check if formula name is listed
    if method in radial_basis_functions.keys():
        RBF = radial_basis_functions[method]
    else:
        raise ValueError("Method {0} not implemented".format(method))

    #-- Creation of data distance matrix
    #-- Data to Data
    if (metric == 'brute'):
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
            metric=metric)
    #-- shape of distance matrix
    N,M = np.shape(Rd)
    #-- if epsilon is not specified
    if epsilon is None:
        #-- calculate norm with mean euclidean distance
        uix,uiy = np.nonzero(np.tri(N,M=M,k=-1))
        epsilon = np.mean(Rd[uix,uiy])

    #-- possible augmentation of the PHI Matrix with polynomial Vectors
    if polynomial is None:
        #-- calculate radial basis function for data-to-data with smoothing
        PHI = RBF(epsilon, Rd) + np.eye(N,M=M)*smooth
        DMAT = zs.copy()
    else:
        #-- number of polynomial coefficients
        nt = (polynomial**2 + 3*polynomial)//2 + 1
        #-- calculate radial basis function for data-to-data with smoothing
        PHI = np.zeros((N+nt,M+nt))
        PHI[:N,:M] = RBF(epsilon, Rd) + np.eye(N,M=M)*smooth
        #-- augmentation of PHI matrix with polynomials
        POLY = polynomial_matrix(xs,ys,polynomial)
        DMAT = np.concatenate(([zs,np.zeros((nt))]),axis=0)
        #-- augment PHI matrix
        for t in range(nt):
            PHI[:N,M+t] = POLY[:,t]
            PHI[N+t,:M] = POLY[:,t]

    #-- Computation of the Weights
    w = np.linalg.lstsq(PHI,DMAT[:,np.newaxis],rcond=-1)[0]

    #-- Computation of distance Matrix
    #-- Computation of distance Matrix (data to mesh points)
    if (metric == 'brute'):
        #-- use linear algebra to compute euclidean distances
        Re = distance_matrix(
            np.array([XI.flatten(),YI.flatten()]),
            np.array([xs,ys])
            )
    else:
        #-- use scipy spatial distance routines
        Rd = scipy.spatial.distance.cdist(
            np.array([XI.flatten(),YI.flatten()]).T,
            np.array([xs, ys]).T,
            metric=metric)
    #-- calculate radial basis function for data-to-mesh matrix
    E = RBF(epsilon,Re)

    #-- possible augmentation of the Evaluation Matrix with polynomial vectors
    if polynomial is not None:
        P = polynomial_matrix(XI.flatten(),YI.flatten(),polynomial)
        E = np.concatenate(([E, P]),axis=1)
    #-- calculate output interpolated array (or matrix)
    if (np.ndim(XI) == 1):
        ZI = np.squeeze(np.dot(E,w))
    else:
        ZI = np.zeros((nx,ny))
        ZI[:,:] = np.dot(E,w).reshape(nx,ny)
    #-- return the interpolated array (or matrix)
    return ZI

#-- define radial basis function formulas
def multiquadric(epsilon, r):
    #-- multiquadratic
    f = np.sqrt((epsilon*r)**2 + 1.0)
    return f

def inverse_multiquadric(epsilon, r):
    #-- inverse multiquadratic
    f = 1.0/np.sqrt((epsilon*r)**2 + 1.0)
    return f

def inverse_quadratic(epsilon, r):
    #-- inverse quadratic
    f = 1.0/(1.0+(epsilon*r)**2)
    return f

def gaussian(epsilon, r):
    #-- gaussian
    f = np.exp(-(epsilon*r)**2)
    return f

def poly_spline1(epsilon, r):
    #-- First-order polyharmonic spline
    f = (epsilon*r)
    return f

def poly_spline3(epsilon, r):
    #-- Third-order polyharmonic spline
    f = (epsilon*r)**3
    return f

def poly_spline5(epsilon, r):
    #-- Fifth-order polyharmonic spline
    f = (epsilon*r)**5
    return f

def thin_plate(epsilon, r):
    #-- thin plate spline
    f = r**2 * np.log(r)
    #-- the spline is zero at zero
    f[r == 0] = 0.0
    return f

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

#-- calculate polynomial matrix to augment radial basis functions
def polynomial_matrix(x,y,order):
    c = 0
    M = len(x)
    N = (order**2 + 3*order)//2 + 1
    POLY = np.zeros((M,N))
    for ii in range(order + 1):
        for jj in range(ii + 1):
            POLY[:,c] = (x**jj)*(y**(ii-jj))
            c += 1
    return POLY
