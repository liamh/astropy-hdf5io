"""
HDF5 group utilities for hierarchical data organization.

Provides convenience functions for saving/loading objects to specific groups
within HDF5 files, supporting nested hierarchies.
"""

import h5py
from fsc.hdf5_io import to_hdf5, from_hdf5

__all__ = [
    'save_to_group',
    'load_from_group',
    'list_groups',
    'print_tree',
    'delete_group',
    'save_recursive',
    'load_recursive',
]


def save_to_group(obj, filename, group_path, mode='a'):
    """
    Save an object to a specific group in an HDF5 file, creating hierarchy as needed.

    This is a convenience wrapper around fsc.hdf5_io for working with group
    hierarchies. Useful for organizing multiple datasets, experiments, or
    time series data in a single HDF5 file.

    Parameters
    ----------
    obj : object
        Object to save (must have HDF5 serialization support via astropy-hdf5io
        or custom to_hdf5() method)
    filename : str
        Path to HDF5 file
    group_path : str
        Path to group, supporting hierarchical paths like 'experiment/run_001/data'
        Leading slash is optional. All intermediate groups are created automatically.
    mode : str, optional
        File mode: 'w' (write/overwrite entire file), 'a' (append/update, default)
        Default is 'a' to preserve existing data.

    Examples
    --------
    Save to a nested hierarchy:

    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> coord = SkyCoord(ra=10*u.degree, dec=40*u.degree, distance=1000*u.pc)
    >>> save_to_group(coord, 'astronomy.hdf5', 'observations/targets/ngc1234')

    Save multiple objects to different parts of hierarchy:

    >>> save_to_group(distance, 'data.hdf5', 'experiment/parameters/distance')
    >>> save_to_group(results, 'data.hdf5', 'experiment/run_001/results')
    >>> save_to_group(metadata, 'data.hdf5', 'experiment/metadata')

    Overwrite entire file:

    >>> save_to_group(data, 'data.hdf5', 'main', mode='w')

    See Also
    --------
    load_from_group : Load object from a specific group
    fsc.hdf5_io.save : Save to root of HDF5 file

    Notes
    -----
    This function uses `h5py.Group.require_group()` which creates all
    intermediate groups in the path if they don't already exist.
    """
    with h5py.File(filename, mode) as f:
        # require_group creates all intermediate groups automatically
        grp = f.require_group(group_path)
        to_hdf5(obj, grp)


def load_from_group(filename, group_path):
    """
    Load an object from a specific group in an HDF5 file.

    Parameters
    ----------
    filename : str
        Path to HDF5 file
    group_path : str
        Path to group, supporting hierarchical paths like 'experiment/run_001/data'
        Leading slash is optional

    Returns
    -------
    object
        Deserialized object

    Examples
    --------
    Load from nested hierarchy:

    >>> coord = load_from_group('astronomy.hdf5', 'observations/targets/ngc1234')

    Load from root-relative path:

    >>> data = load_from_group('data.hdf5', '/experiment/run_001/results')

    See Also
    --------
    save_to_group : Save object to a specific group
    fsc.hdf5_io.load : Load from root of HDF5 file

    Raises
    ------
    KeyError
        If the specified group path does not exist in the file
    """
    with h5py.File(filename, 'r') as f:
        return from_hdf5(f[group_path])


def list_groups(filename, group_path='/'):
    """
    List all groups and datasets in a specific group.

    Parameters
    ----------
    filename : str
        Path to HDF5 file
    group_path : str, optional
        Path to group to list (default: '/' for root)

    Returns
    -------
    dict
        Dictionary with 'groups' and 'datasets' keys, each containing lists
        of names (not full paths)

    Examples
    --------
    List root contents:

    >>> list_groups('data.hdf5')
    {'groups': ['experiment', 'backup'], 'datasets': ['metadata']}

    List specific group:

    >>> list_groups('data.hdf5', 'experiment')
    {'groups': ['run_001', 'run_002'], 'datasets': []}

    See Also
    --------
    print_tree : Print hierarchical structure of entire file or group
    """
    with h5py.File(filename, 'r') as f:
        grp = f[group_path]
        groups = [key for key in grp.keys() if isinstance(grp[key], h5py.Group)]
        datasets = [key for key in grp.keys() if isinstance(grp[key], h5py.Dataset)]
        return {'groups': groups, 'datasets': datasets}


def print_tree(filename, group_path='/', indent=0, max_depth=None):
    """
    Print the hierarchical structure of an HDF5 file.

    Displays the group hierarchy and datasets in a tree format, making it
    easy to visualize the organization of data in the file.

    Parameters
    ----------
    filename : str
        Path to HDF5 file
    group_path : str, optional
        Path to group to start from (default: '/' for root)
    indent : int, optional
        Initial indentation level (used internally for recursion)
    max_depth : int, optional
        Maximum depth to traverse (None for unlimited)

    Examples
    --------
    Print entire file structure:

    >>> print_tree('astronomy.hdf5')
    /
      observations/
        targets/
          ngc1234/
            coordinate [astropy_hdf5io.SkyCoord]
          ngc5678/
        metadata [astropy_hdf5io.Time]

    Print only a specific subtree:

    >>> print_tree('data.hdf5', 'experiment/run_001')
    /experiment/run_001/
      parameters/
      results/
        output [astropy_hdf5io.Quantity]

    Limit depth:

    >>> print_tree('data.hdf5', max_depth=2)

    See Also
    --------
    list_groups : Get list of groups and datasets programmatically
    """
    with h5py.File(filename, 'r') as f:
        _print_group(f[group_path], indent, max_depth, current_depth=0)


def _print_group(group, indent=0, max_depth=None, current_depth=0):
    """Helper function for print_tree."""
    prefix = '  ' * indent
    print(f"{prefix}{group.name}/")

    if max_depth is not None and current_depth >= max_depth:
        return

    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            _print_group(item, indent + 1, max_depth, current_depth + 1)
        else:  # Dataset
            # Try to show the type_tag if available
            # Datasets are accessed differently than groups
            type_tag = ''
            try:
                # For datasets, we need to check if 'type_tag' exists as a sub-item
                if 'type_tag' in item:
                    type_tag_data = item['type_tag'][()]
                    if isinstance(type_tag_data, bytes):
                        type_tag = type_tag_data.decode('utf-8')
                    else:
                        type_tag = str(type_tag_data)
            except (KeyError, ValueError, TypeError):
                # If we can't get the type_tag, just skip it
                pass

            type_str = f" [{type_tag}]" if type_tag else ""
            print(f"{'  ' * (indent + 1)}{key}{type_str}")


def delete_group(filename, group_path):
    """
    Delete a group from an HDF5 file.

    Warning: This permanently deletes the group and all its contents!

    Parameters
    ----------
    filename : str
        Path to HDF5 file
    group_path : str
        Path to group to delete

    Examples
    --------
    Delete a backup:

    >>> delete_group('data.hdf5', 'experiment/backup')

    Delete old results:

    >>> delete_group('data.hdf5', 'experiment/run_000')

    Notes
    -----
    The deletion is immediate and cannot be undone. Make sure you have
    backups if the data is important.

    Raises
    ------
    KeyError
        If the specified group does not exist
    """
    with h5py.File(filename, 'a') as f:
        if group_path in f:
            del f[group_path]
        else:
            raise KeyError(f"Group '{group_path}' not found in {filename}")

def save_recursive(obj, filename, base_path='/', mode='a', skip_types=None):
    """
    Recursively save a nested dictionary/Munch structure to HDF5 groups.

    Traverses a nested dict-like structure and saves each serializable object
    to HDF5, using the nesting structure as the group hierarchy.

    Parameters
    ----------
    obj : dict, Munch, or similar
        Nested dictionary-like object to save. Keys become group names.
    filename : str
        Path to HDF5 file
    base_path : str, optional
        Base group path to save under (default: '/' for root)
    mode : str, optional
        File mode: 'w' (write/overwrite), 'a' (append, default)
    skip_types : list of type, optional
        List of types to skip (won't traverse or save). Useful for
        skipping generator objects, functions, etc.

    Returns
    -------
    dict
        Summary of what was saved: {group_path: type_name, ...}

    Examples
    --------
    Save a Munch structure with nested orbit data:

    >>> from munch import Munch
    >>> demoa = Munch()
    >>> demoa.init = Munch()
    >>> demoa.init.pv = np.array([5740132.6835, 3314067.15, 0.0, ...])
    >>> demoa.init.pvt = pvtcart(demoa.init.pv, time)
    >>> demoa.propn = Munch()
    >>> demoa.propn.pvt = propagated_states
    >>>
    >>> save_recursive(demoa, 'mission.h5', 'satellites/demoa')
    >>> # Creates: satellites/demoa/init/pv
    >>> #          satellites/demoa/init/pvt
    >>> #          satellites/demoa/propn/pvt

    Skip certain object types:

    >>> save_recursive(demoa, 'mission.h5', skip_types=[type(gen)])

    See Also
    --------
    load_recursive : Load a nested structure from HDF5
    save_to_group : Save a single object to a group

    Notes
    -----
    - Objects are only saved if they have HDF5 serialization support
    - Dict-like objects (dict, Munch, etc.) are traversed but not saved
    - Non-serializable objects are skipped with a warning
    - Group names are sanitized to be HDF5-compatible
    """
    from collections.abc import Mapping

    if skip_types is None:
        skip_types = []

    saved_items = {}

    def _is_dict_like(obj):
        """Check if object is dict-like (has keys and values)."""
        return isinstance(obj, Mapping) or (hasattr(obj, '__dict__') and not hasattr(obj, 'to_hdf5'))

    def _get_dict_items(obj):
        """Get items from dict-like object."""
        if isinstance(obj, Mapping):
            return obj.items()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__.items()
        else:
            return []

    def _sanitize_name(name):
        """Sanitize name to be HDF5-compatible."""
        sanitized = str(name).replace(' ', '_').replace('/', '_')
        if sanitized and sanitized[0].isdigit():
            sanitized = 'n' + sanitized
        return sanitized

    def _traverse_and_save(obj, parent_group, current_path, base_len):
        """Recursively traverse object and save to HDF5."""
        if any(isinstance(obj, skip_type) for skip_type in skip_types):
            return

        if _is_dict_like(obj):
            for key, value in _get_dict_items(obj):
                if isinstance(key, str) and key.startswith('_'):
                    continue
                if callable(value) and not hasattr(value, 'to_hdf5'):
                    continue

                sanitized_key = _sanitize_name(key)
                new_path = f"{current_path}/{sanitized_key}" if current_path else sanitized_key

                _traverse_and_save(value, parent_group, new_path, base_len)
        else:
            try:
                path_parts = [p for p in current_path.split('/') if p]

                if len(path_parts) > 0:
                    current_grp = parent_group
                    for part in path_parts[:-1]:
                        if part not in current_grp:
                            current_grp = current_grp.create_group(part)
                        else:
                            current_grp = current_grp[part]

                    obj_name = path_parts[-1]
                    if obj_name in current_grp:
                        del current_grp[obj_name]
                    obj_group = current_grp.create_group(obj_name)
                else:
                    obj_group = parent_group

                to_hdf5(obj, obj_group)

                # Record the path relative to base_path (without leading /)
                relative_path = current_path[base_len:].lstrip('/')
                type_name = f"{type(obj).__module__}.{type(obj).__name__}"
                saved_items[relative_path] = type_name

            except Exception as e:
                import warnings
                warnings.warn(
                    f"Could not save object at '{current_path}' "
                    f"(type: {type(obj).__name__}): {e}"
                )

    with h5py.File(filename, mode) as f:
        if base_path == '/' or base_path == '':
            start_group = f
            start_path = ''
            base_len = 0
        else:
            base_parts = [p for p in base_path.split('/') if p]
            start_group = f
            for part in base_parts:
                if part not in start_group:
                    start_group = start_group.create_group(part)
                else:
                    start_group = start_group[part]
            start_path = '/'.join(base_parts)
            base_len = len(start_path)

        _traverse_and_save(obj, start_group, start_path, base_len)

    return saved_items

def load_recursive(filename, base_path='/', reconstruct_dicts=True):
    """
    Recursively load a nested HDF5 group structure.

    Loads all objects from nested HDF5 groups and reconstructs them
    as a nested dictionary structure matching the original hierarchy.

    Parameters
    ----------
    filename : str
        Path to HDF5 file
    base_path : str, optional
        Base group path to load from (default: '/' for root)
    reconstruct_dicts : bool, optional
        If True (default), return nested dicts matching group structure.
        If False, return flat dict with full paths as keys.

    Returns
    -------
    dict
        Nested dictionary containing loaded objects, or flat dict of paths

    Examples
    --------
    Load entire nested structure:

    >>> data = load_recursive('mission.h5', 'satellites/demoa')
    >>> data['init']['pvt']  # Access nested data

    Load as flat dictionary:

    >>> data = load_recursive('mission.h5', reconstruct_dicts=False)
    >>> data['satellites/demoa/init/pvt']  # Access by full path

    Convert to Munch for attribute access:

    >>> from munch import munchify
    >>> data = load_recursive('mission.h5', 'satellites/demoa')
    >>> demoa = munchify(data)
    >>> demoa.init.pvt  # Attribute-style access

    See Also
    --------
    save_recursive : Save a nested structure to HDF5
    load_from_group : Load a single object from a group

    Notes
    -----
    Only objects with HDF5 deserialization support can be loaded.
    Groups that don't contain serialized objects are represented as
    empty dicts in the output.
    """
    def _reconstruct_nested_dict(flat_dict):
        """Convert flat dict with paths to nested dict."""
        nested = {}

        for path, obj in flat_dict.items():
            parts = [p for p in path.split('/') if p]

            current = nested
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            if parts:
                current[parts[-1]] = obj

        return nested

    def _load_group(group, current_path='', strip_prefix=''):
        """Recursively load all items from a group."""
        loaded = {}

        for key in group.keys():
            item = group[key]
            item_path = f"{current_path}/{key}" if current_path else key

            if isinstance(item, h5py.Group):
                if 'type_tag' in item:
                    try:
                        obj = from_hdf5(item)
                        # Strip the prefix if specified
                        save_path = item_path
                        if strip_prefix and item_path.startswith(strip_prefix):
                            save_path = item_path[len(strip_prefix):].lstrip('/')
                        loaded[save_path] = obj
                    except Exception as e:
                        import warnings
                        warnings.warn(
                            f"Could not load object at '{item_path}': {e}"
                        )
                else:
                    loaded.update(_load_group(item, item_path, strip_prefix))

        return loaded

    with h5py.File(filename, 'r') as f:
        if base_path == '/' or base_path == '':
            start_group = f
            strip_prefix = ''
        else:
            start_path_normalized = base_path.lstrip('/').rstrip('/')
            if start_path_normalized in f:
                start_group = f[start_path_normalized]
                # When loading from a base_path, strip it from result paths
                strip_prefix = start_path_normalized
            else:
                raise KeyError(f"Group '{base_path}' not found in {filename}")

        loaded_items = _load_group(start_group, '', strip_prefix)

    if reconstruct_dicts:
        return _reconstruct_nested_dict(loaded_items)
    else:
        return loaded_items
