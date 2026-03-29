"""
AstroPy serialization support for fsc.hdf5-io
"""

__version__ = "0.1.0"

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

# Munch support
from ._munch_utils import load_munch

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
    # Munch utilities
    'load_munch',
]
