"""Tests for astropy.time.Time serialization"""

import pytest
import numpy as np
from astropy.time import Time
import astropy.units as u
from fsc.hdf5_io import save, load
from tempfile import NamedTemporaryFile

import astropy_hdf5io


class TestTime:
    """Tests for Time serialization"""

    def test_scalar_time(self):
        """Test scalar Time object"""
        t = Time('2023-01-15T12:30:45')

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, Time)
        assert np.isclose(loaded.jd, t.jd)
        assert loaded.scale == t.scale

    def test_array_time(self):
        """Test array of Time objects"""
        times = Time(['2023-01-01', '2023-01-02', '2023-01-03'])

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(times, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, Time)
        assert len(loaded) == len(times)
        assert np.allclose(loaded.jd, times.jd)

    def test_time_jd_format(self):
        """Test Time with Julian Date format"""
        t = Time(2459580.5, format='jd')

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        assert np.isclose(loaded.jd, t.jd)

    def test_time_mjd_format(self):
        """Test Time with Modified Julian Date format"""
        t = Time(59580.0, format='mjd')

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        assert np.isclose(loaded.mjd, t.mjd)

    def test_time_iso_format(self):
        """Test Time with ISO format"""
        t = Time('2023-01-15 12:30:45', format='iso', scale='utc')

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        assert loaded.iso == t.iso
        assert loaded.scale == t.scale

    def test_time_different_scales(self):
        """Test Time with different time scales"""
        scales = ['utc', 'tai', 'tt', 'tcg', 'tcb', 'tdb']

        for scale in scales:
            t = Time('2023-01-15T12:00:00', scale=scale)

            with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
                save(t, f.name)
                loaded = load(f.name)

            assert loaded.scale == t.scale
            assert np.isclose(loaded.jd, t.jd)

    def test_time_from_datetime(self):
        """Test Time created from datetime"""
        from datetime import datetime
        dt = datetime(2023, 1, 15, 12, 30, 45)
        t = Time(dt)
