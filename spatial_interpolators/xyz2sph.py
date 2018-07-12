#!/usr/bin/env python
u"""
xyz2sph.py
Written by Tyler Sutterley (UPDATED 08/2016)

Converts Cartesian coordinates to spherical coordinates

CALLING SEQUENCE:
	sph = xyz2sph(x,y,z)
	lon = sph['lon']
	lat = sph['lat']
	rad = sph['rad']

INPUTS:
	x,y,z in cartesian coordinates
OUTPUTS:
	lon: spherical longitude
	lat: spherical latitude
	rad: spherical radius
"""

def xyz2sph(x,y,z):
	import numpy as np

	#-- calculate radius
	rad = np.sqrt(x**2.0 + y**2.0 + z**2.0)

	#-- calculate angular coordinates
	#-- phi: azimuthal angle
	phi = np.arctan2(y,x)
	#-- th: polar angle
	th = np.arccos(z/rad)

	#-- convert to degrees and fix to 0:360
	lon = 180.0*phi/np.pi
	ii = np.nonzero(lon < 0)
	count = np.count_nonzero(lon < 0)
	if (count != 0):
		lon[ii] = lon[ii]+360.0
	#-- convert to degrees and fix to -90:90
	lat = 90.0 -(180.0*th/np.pi)

	return {'lon': lon, 'lat':lat, 'rad':rad}
