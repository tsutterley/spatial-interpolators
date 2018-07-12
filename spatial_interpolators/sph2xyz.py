#!/usr/bin/env python
u"""
sph2xyz.py
Written by Tyler Sutterley (07/2013)

Converts spherical coordinates to Cartesian coordinates

CALLING SEQUENCE:
	xyz = sph2xyz(lon,lat,RAD=6371.0)
	x = xyz['x']
	y = xyz['y']
	z = xyz['z']

INPUTS
	lon: spherical longitude
	lat: spherical latitude
OUTPUTS:
	x,y,z in cartesian coordinates
OPTIONS:
	RAD: radius (default is mean Earth radius)

PYTHON DEPENDENCIES:
	numpy: Scientific Computing Tools For Python (http://www.numpy.org)
"""

def sph2xyz(lon,lat,RAD=6371.0):
	import numpy as np

	ilon = np.nonzero(lon < 0)
	count = np.count_nonzero(lon < 0)
	if (count != 0):
		lon[ilon] = lon[ilon]+360.0

	phi = np.pi*lon/180.0
	th = np.pi*(90.0 - lat)/180.0

	x=RAD*np.sin(th)*np.cos(phi)#-- x
	y=RAD*np.sin(th)*np.sin(phi)#-- y
	z=RAD*np.cos(th)#-- z

	return {'x':x,'y':y,'z':z}
