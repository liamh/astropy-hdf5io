"""Munch support for astropy-hdf5io."""

from munch import munchify
from fsc.hdf5_io import load as _load


def load_munch(filename):
    """
    Load an HDF5 file and return the result as a Munch.

    Equivalent to ``munchify(load(filename))``. Nested dicts are
    recursively converted, so attribute-style access works at all levels.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.

    Returns
    -------
    munch.Munch

    Examples
    --------
    >>> import astropy_hdf5io
    >>> demoa = astropy_hdf5io.load_munch('demoa.h5')
    >>> demoa.init.pvt   # attribute-style access
    """
    return munchify(_load(filename))