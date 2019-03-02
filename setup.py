from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Distutils import build_ext
setup(
	name='spatial-interpolators',
	version='1.0.0.2',
	description='Spatial interpolation tools for Python',
	url='https://github.com/tsutterley/spatial-interpolators',
	author='Tyler Sutterley',
	author_email='tyler.c.sutterley@nasa.gov',
	license='MIT',
	classifiers=[
		'Development Status :: 3 - Alpha',
		'Intended Audience :: Science/Research',
		'Topic :: Scientific/Engineering :: Physics',
		'License :: OSI Approved :: MIT License',
		'Programming Language :: Python :: 2',
		'Programming Language :: Python :: 2.7',
	],
	keywords='spatial interpolation, regridding, regridding over a sphere',
	packages=find_packages(),
	install_requires=['numpy','scipy','cython','matplotlib'],
	cmdclass={'build_ext': build_ext},
	ext_modules=[Extension("PvQv_C",["spatial_interpolators/PvQv_C.pyx"])],
)
