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