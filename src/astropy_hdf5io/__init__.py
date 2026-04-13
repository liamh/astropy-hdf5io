"""
AstroPy serialization support for fsc.hdf5-io
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("astropy-hdf5io")
except PackageNotFoundError:
    # Package is not installed (e.g. running directly from source)
    __version__ = "unknown"

# Import all serializers to register them
from . import _quantity
from . import _skycoord
from . import _time
from . import _coordinates
from . import _table

# Import group utilities
from ._group_utils import (
    save_to_group,
    load_from_group,
    list_groups,
    print_tree,
    delete_group,
    save_recursive,
    load_recursive,
)

# Munch support (optional — requires 'munch' package)
try:
    from ._munch_utils import load_munch
except ImportError:
    def load_munch(filename):
        raise ImportError(
            "The 'munch' package is required for load_munch(). "
            "Install it with: pip install astropy-hdf5io[munch]"
        )

__all__ = [
    '__version__',
    # Group utilities
    'save_to_group',
    'load_from_group',
    'list_groups',
    'print_tree',
    'delete_group',
    # Recursive utilities
    'save_recursive',
    'load_recursive',
    # Munch utilities (only if munch is installed)
    'load_munch',
]
