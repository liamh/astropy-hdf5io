"""Munch support for astropy-hdf5io."""

from fsc.hdf5_io import load as _load


def load_munch(filename):
    """
    Load an HDF5 file and return the result as a Munch.

    Equivalent to ``munchify(load(filename))``. Nested dicts are
    recursively converted, so attribute-style access works at all levels.

    Requires the optional ``munch`` package::

        pip install astropy-hdf5io[munch]

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
    try:
        from munch import munchify
    except ImportError as e:
        raise ImportError(
            "The 'munch' package is required for load_munch(). "
            "Install it with: pip install astropy-hdf5io[munch]"
        ) from e

    return munchify(_load(filename))