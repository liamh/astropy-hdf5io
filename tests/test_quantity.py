"""Tests for Quantity serialization"""

import pytest
import numpy as np
import astropy.units as u
from fsc.hdf5_io import save, load
from tempfile import NamedTemporaryFile

# Import to register serializers
import astropy_hdf5io


def test_scalar_quantity():
    """Test saving and loading scalar Quantity"""
    distance = 1171 * u.Mpc

    with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
        save(distance, f.name)
        loaded = load(f.name)

    assert isinstance(loaded, u.Quantity)
    assert loaded.value == distance.value
    assert loaded.unit == distance.unit


def test_array_quantity():
    """Test saving and loading array Quantity"""
    distances = np.array([100, 200, 300]) * u.kpc

    with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
        save(distances, f.name)
        loaded = load(f.name)

    assert isinstance(loaded, u.Quantity)
    assert np.allclose(loaded.value, distances.value)
    assert loaded.unit == distances.unit


def test_quantity_in_dict():
    """Test Quantity nested in dictionary"""
    data = {
        'distance': 778 * u.kpc,
        'velocity': 301 * u.km / u.s,
        'name': 'Andromeda'
    }

    with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
        save(data, f.name)
        loaded = load(f.name)

    assert loaded['distance'].value == data['distance'].value
    assert loaded['distance'].unit == data['distance'].unit
    assert loaded['velocity'].value == data['velocity'].value
    assert loaded['velocity'].unit == data['velocity'].unit
    assert loaded['name'] == data['name']


def test_quantity_in_list():
    """Test Quantity nested in list"""
    data = [100 * u.Mpc, 200 * u.Mpc, 300 * u.Mpc]

    with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
        save(data, f.name)
        loaded = load(f.name)

    assert len(loaded) == len(data)
    for orig, load_val in zip(data, loaded):
        assert load_val.value == orig.value
        assert load_val.unit == orig.unit


def test_complex_units():
    """Test complex unit combinations"""
    acceleration = 9.8 * u.m / u.s**2

    with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
        save(acceleration, f.name)
        loaded = load(f.name)

    assert loaded.value == acceleration.value
    assert loaded.unit == acceleration.unit


def test_equivalencies():
    """Test that unit equivalencies are preserved"""
    energy = 1.0 * u.eV

    with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
        save(energy, f.name)
        loaded = load(f.name)

    # Should be able to convert to equivalent units
    loaded_joules = loaded.to(u.J)
    energy_joules = energy.to(u.J)

    assert np.isclose(loaded_joules.value, energy_joules.value)


def test_structured_quantity():
    """Test saving and loading a Quantity backed by a structured/record array"""
    dtype = np.dtype([('r', float), ('lon', float), ('lat', float)])
    arr = np.array([(6371.0, 45.0, 30.0)], dtype=dtype)
    q = u.Quantity(arr, unit=u.StructuredUnit('km,deg,deg', names=('r', 'lon', 'lat')))

    with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
        save(q, f.name)
        loaded = load(f.name)

    assert isinstance(loaded, u.Quantity)
    assert loaded.unit.field_names == q.unit.field_names
    assert loaded.unit['r'] == q.unit['r']
    assert loaded.unit['lon'] == q.unit['lon']
    assert loaded.unit['lat'] == q.unit['lat']
    assert loaded.value['r'][0] == q.value['r'][0]
    assert loaded.value['lon'][0] == q.value['lon'][0]
    assert loaded.value['lat'][0] == q.value['lat'][0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
