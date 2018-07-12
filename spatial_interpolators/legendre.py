#!/usr/bin/env python
u"""
legendre.py
Computes the associated Legendre functions of degree N and order M = 0:N
	evaluated for each element X.
N must be a scalar integer and X must contain real values between -1 <= X <= 1.

Program is based on a Fortran program by Robert L. Parker, Scripps Institution
	of Oceanography, Institute for Geophysics and Planetary Physics, UCSD. 1993.
Parallels the MATLAB legendre function

PYTHON DEPENDENCIES:
	numpy: Scientific Computing Tools For Python (http://www.numpy.org)

REFERENCES:
	M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
		Dover Publications, 1965, Ch. 8.
	J. A. Jacobs, "Geomagnetism", Academic Press, 1987, Ch.4.
UPDATE HISTORY:
	Written 08/2016
"""
import numpy as np

def legendre(n,x):
	#-- Convert x to a single row vector
	x = np.squeeze(x).flatten()

	#-- for the n = 0 case
	if (n == 0):
		Pl = np.ones((1,len(x)), dtype=np.float)
		return Pl

	#-- for all other degrees
	rootn = np.sqrt(np.arange(0,2*n+1))#-- +1 to include 2*n
	s = np.sqrt(1 - x**2)
	P = np.zeros((n+3,len(x)), dtype=np.float)

	#-- Calculate TWOCOT
	twocot = -2.0*x/s

	#-- Find values of x,s for which there will be underflow
	sn = (-s)**n
	tol = np.sqrt(np.finfo(np.float).tiny)
	count = np.count_nonzero((s > 0) & (np.abs(sn) <= tol))
	if (count > 0):
		ind, = np.nonzero((s > 0) & (np.abs(sn) <= tol))
		#-- Approx solution of x*ln(x) = Pl
		v = 9.2 - np.log(tol)/(n*s[ind])
		w = 1.0/np.log(v)
		m1 = 1+n*s[ind]*v*w*(1.0058+ w*(3.819 - w*12.173))
		m1 = np.min([n, np.floor(m1)]).astype(np.int)
		#-- Column-by-column recursion
		for k,mm1 in enumerate(m1):
			col = ind[k]
			P[mm1-1:n+1,col] = 0.0
			#-- Start recursion with proper sign
			tstart = np.finfo(np.float).eps
			P[mm1-1,col] = np.sign(np.fmod(mm1,2)-0.5)*tstart
			if (x[col] < 0):
				P[mm1-1,col] = np.sign(np.fmod(n+1,2)-0.5)*tstart
			#-- Recur from m1 to m = 0, accumulating normalizing factor.
			sumsq = tol.copy()
			for m in range(mm1-2,-1,-1):
				P[m,col] = ((m+1)*twocot[col]*P[m+1,col] - \
					rootn[n+m+2]*rootn[n-m-1]*P[m+2,col]) / \
					(rootn[n+m+1]*rootn[n-m])
				sumsq += P[m,col]**2
			#-- calculate scale
			scale = 1.0/np.sqrt(2.0*sumsq - P[0,col]**2)
			P[0:mm1+1,col] = scale*P[0:mm1+1,col]

	#-- Find the values of x,s for which there is no underflow, and for
	#-- which twocot is not infinite (x~=1).
	count = np.count_nonzero((x != 1) & (np.abs(sn) >= tol))
	if (count > 0):
		nind, = np.nonzero((x != 1) & (np.abs(sn) >= tol))
		#-- Produce normalization constant for the m = n function
		d = np.arange(2,2*n+2,2)
		c = np.prod(1.0 - 1.0/d)
		#-- Use sn = (-s)**n (written above) to write the m = n function
		P[n,nind] = np.sqrt(c)*sn[nind]
		P[n-1,nind] = P[n,nind]*twocot[nind]*n/rootn[-1]

		#-- Recur downwards to m = 0
		for m in range(n-2,-1,-1):
			P[m,nind] = (P[m+1,nind]*twocot[nind]*(m+1) - \
				P[m+2,nind]*rootn[n+m+2]*rootn[n-m-1])/(rootn[n+m+1]*rootn[n-m])

	#-- calculate Pl from P
	Pl = P[0:n+1,:]

	#-- Polar argument (x = +-1)
	count = np.count_nonzero(s == 0)
	if (count > 0):
		s0, = np.nonzero(s == 0)
		Pl[0,s0] = x[s0]**n

	#-- Calculate the standard A&S functions (i.e., unnormalized) by
	#-- multiplying each row by: sqrt((n+m)!/(n-m)!) = sqrt(prod(n-m+1:n+m))
	for m in range(1,n):
		Pl[m,:] = np.prod(rootn[n-m+1:n+m+1])*Pl[m,:]

	#-- Last row (m = n) must be done separately to handle 0!.
	#-- NOTE: the coefficient for (m = n) overflows for n>150.
	Pl[n,:] = np.prod(rootn[1:])*Pl[n,:]

	return Pl
