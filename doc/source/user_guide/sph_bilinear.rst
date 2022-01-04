===============
sph_bilinear.py
===============

- Interpolates data over a sphere using bilinear functions

Calling Sequence
################

.. code-block:: python

    import spatial_interpolators as spi
    zi = spi.sph_bilinear(x,y,z,xi,yi)

`Source code`__

.. __: https://github.com/tsutterley/spatial-interpolators/blob/master/spatial_interpolators/sph_bilinear.py


Arguments
#########

1. ``x``: input longitude
2. ``y``: input latitude
3. ``z``: input data
4. ``xi``: output longitude
5. ``yi``: output latitude

Keyword arguments
#################

- ``flattened``: input xi, yi are flattened arrays
- ``fill_value``: invalid value

Returns
#######

- ``zi``: interpolated data
