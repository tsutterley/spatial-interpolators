spatial-interpolators
=====================

[![Language](https://img.shields.io/badge/python-v3.7-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/tsutterley/spatial-interpolators/blob/master/LICENSE)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tsutterley/spatial-interpolators/master)
[![Binder](https://binder.pangeo.io/badge.svg)](https://binder.pangeo.io/v2/gh/tsutterley/spatial-interpolators/master)

Functions to spatially interpolate data over Cartesian and spherical grids

##### `barnes_objective.py`
Barnes objective analysis for the optimal interpolation of an input grid using
	a successive corrections scheme
	[(Barnes, 1994)](https://doi.org/10.1175/1520-0426%281994%29011<1433:AOTBOA>2.0.CO;2).

##### `biharmonic_spline.py`
Interpolates a sparse grid using 2D biharmonic splines
	[(Sandwell, 1987)](https://doi.org/10.1029/GL014i002p00139)
	with or without tension parameters
	[(Wessel and Bercovici, 1998)](https://doi.org/10.1023/A:1021713421882)
	or using the regularized function of
	[Mitasova and Mitas (1993)](https://doi.org/10.1007/BF00893171).

##### `sph_spline.py`
Interpolates a sparse grid over a sphere using spherical surface splines in
	tension following [Wessel and Becker (2008)](10.1111/j.1365-246X.2008.03829.x).

##### `radial_basis.py`
Interpolates a sparse grid using radial basis functions
	[(Hardy, 1971)](https://doi.org/10.1029/JB076i008p01905).

##### `compact_radial_basis.py`
Interpolates a sparse grid using compactly supported radial basis functions
	[(Wendland, 1995)](https://doi.org/10.1007/BF02123482).

##### `sph_radial_basis.py`
Interpolates a sparse grid over a sphere using radial basis functions with
	QR factorization option to eliminate ill-conditioning
	[(Fornberg and Piret, 2007)](https://doi.org/10.1137/060671991) and
	[(Fornberg et al., 2011)](https://doi.org/10.1137/09076756X).

##### `shepard_interpolant.py`
Evaluates Shepard interpolants to 2D data based on inverse distance weighting.

##### `sph_bilinear.py`
Spherical interpolation routine for gridded data using bilinear interpolation.

##### `interpolate_franke.ipynb`
Test notebook for visualizing the different Cartesian interpolators

##### `interpolate_sphere.ipynb`
Test notebook for visualizing the different spherical interpolators

#### Dependencies
- [numpy: Scientific Computing Tools For Python](https://numpy.org)
- [scipy: Scientific Tools for Python](https://docs.scipy.org/doc//)
- [cython: C-extensions for Python](http://cython.org/)
- [matplotlib: Python 2D plotting library](https://matplotlib.org/)

#### Download
The program homepage is:  
https://github.com/tsutterley/spatial-interpolators  
A zip archive of the latest version is available directly at:  
https://github.com/tsutterley/spatial-interpolators/archive/master.zip  

#### Disclaimer
This program is not sponsored or maintained by the Universities Space Research Association (USRA) or NASA.
It is provided here for your convenience but _with no guarantees whatsoever_.

#### License
The content of this project is licensed under the [Creative Commons Attribution 4.0 Attribution license](https://creativecommons.org/licenses/by/4.0/) and the source code is licensed under the [MIT license](LICENSE).
