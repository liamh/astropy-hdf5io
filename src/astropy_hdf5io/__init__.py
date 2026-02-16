"""
AstroPy serialization support for fsc.hdf5-io

This package provides automatic HDF5 serialization for AstroPy types
when using fsc.hdf5-io.
"""

__version__ = "0.1.0"


__all__ = ['__version__']



"""
AstroPy serialization support for fsc.hdf5-io

This package provides automatic HDF5 serialization for AstroPy types
when using fsc.hdf5-io.

Usage:
    Simply import both packages and serialization will work automatically:

    >>> import astropy.units as u
    >>> from fsc.hdf5_io import save, load
    >>> import astropy_hdf5io  # Registers serializers
    >>>
    >>> distance = 1171 * u.Mpc
    >>> save(distance, 'distance.hdf5')
    >>> loaded = load('distance.hdf5')
"""

__version__ = "0.1.0"

# Import all serializers to register them
from . import _quantity
from . import _skycoord
from . import _time
from . import _coordinates
from . import _table

__all__ = ['__version__']

# The entry point system will import this module automatically when
# fsc.hdf5_io.load() is called, so users don't even need to import it explicitly!
