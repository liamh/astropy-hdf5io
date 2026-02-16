"""Tests for SkyCoord serialization"""

import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from fsc.hdf5_io import save, load
from tempfile import NamedTemporaryFile

import astropy_hdf5io


def test_skycoord_2d():
    """Test SkyCoord without distance"""
    coord = SkyCoord(ra=10*u.degree, dec=20*u.degree, frame='icrs')

    with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
        save(coord, f.name)
        loaded = load(f.name)

    assert isinstance(loaded, SkyCoord)
    assert loaded.ra.value == coord.ra.value
    assert loaded.dec.value == coord.dec.value


def test_skycoord_3d():
    """Test SkyCoord with distance"""
    coord = SkyCoord(ra=10*u.degree, dec=20*u.degree,
                     distance=100*u.pc, frame='icrs')

    with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
        save(coord, f.name)
        loaded = load(f.name)

    assert loaded.ra.value == coord.ra.value
    assert loaded.dec.value == coord.dec.value
    assert loaded.distance.value == coord.distance.value


def test_skycoord_in_dict():
    """Test SkyCoord nested in dictionary"""
    data = {
        'target': SkyCoord(ra=83.633*u.degree, dec=22.014*u.degree),
        'name': 'M42'
    }

    with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
        save(data, f.name)
        loaded = load(f.name)

    assert loaded['target'].ra.value == data['target'].ra.value
    assert loaded['name'] == data['name']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
