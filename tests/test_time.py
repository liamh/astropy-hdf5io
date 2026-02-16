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

    def test_time_format_preservation_isot(self):
        """Test that ISOT format is preserved through serialization"""
        t = Time('2023-01-01T12:30:45', format='isot')

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        assert loaded.format == 'isot'
        assert loaded.scale == t.scale
        assert loaded == t

    def test_time_format_preservation_iso(self):
        """Test that ISO format is preserved through serialization"""
        t = Time('2023-01-01 12:30:45', format='iso')

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        assert loaded.format == 'iso'
        assert loaded == t

    def test_time_format_preservation_jd(self):
        """Test that JD format is preserved through serialization"""
        t = Time(2459945.5, format='jd')

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        assert loaded.format == 'jd'
        assert np.isclose(loaded.jd, t.jd)

    def test_time_format_preservation_datetime(self):
        """Test that datetime format is preserved"""
        from datetime import datetime
        dt = datetime(2023, 1, 1, 12, 30, 45)
        t = Time(dt, format='datetime')

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        assert loaded.format == 'datetime'
        assert loaded == t

    def test_time_precision_preservation(self):
        """Test that precision is preserved for ISOT format"""
        t = Time('2023-01-01T12:30:45.123456', format='isot', precision=6)

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        assert loaded.format == 'isot'
        assert loaded.precision == 6
        assert str(loaded) == str(t)

    def test_time_precision_default(self):
        """Test time with default precision"""
        t = Time('2023-01-01T12:30:45', format='isot')

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        assert loaded.format == 'isot'
        # Default precision is 3 for ISOT
        assert loaded == t

    def test_time_array_format_preservation(self):
        """Test that format is preserved for time arrays"""
        times = Time(['2023-01-01T00:00:00',
                      '2023-01-01T06:00:00',
                      '2023-01-01T12:00:00'], format='isot')

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(times, f.name)
            loaded = load(f.name)

        assert loaded.format == 'isot'
        assert len(loaded) == len(times)
        assert all(loaded == times)

    def test_time_jd_storage(self):
        """Test that time is stored using two-part JD internally for maximum precision"""
        import h5py
        t = Time('2023-01-01T12:00:00', format='isot')

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)

            # Check that jd1 and jd2 are used for storage (two-part representation)
            with h5py.File(f.name, 'r') as hf:
                assert 'jd1' in hf
                assert 'jd2' in hf
                assert 'format' in hf.attrs
                assert hf.attrs['format'] == 'isot'

            # Verify it still loads correctly
            loaded = load(f.name)
            assert loaded.format == 'isot'
            assert loaded == t

class TestTimeEdgeCases:
    """Test edge cases and special scenarios for Time serialization"""

    def test_time_with_location(self):
        """Test Time with EarthLocation"""
        from astropy.coordinates import EarthLocation

        loc = EarthLocation.of_site('greenwich')
        t = Time('2023-01-01T12:00:00', format='isot', location=loc)

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        assert loaded.format == 'isot'
        assert loaded.location is not None
        assert np.allclose(loaded.location.x.value, loc.x.value)
        assert np.allclose(loaded.location.y.value, loc.y.value)
        assert np.allclose(loaded.location.z.value, loc.z.value)

    def test_time_scalar_vs_array(self):
        """Test that both scalar and array times work"""
        t_scalar = Time('2023-01-01T12:00:00', format='isot')
        t_array = Time(['2023-01-01T12:00:00', '2023-01-02T12:00:00'], format='isot')

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t_scalar, f.name)
            loaded_scalar = load(f.name)

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t_array, f.name)
            loaded_array = load(f.name)

        assert loaded_scalar.isscalar
        assert not loaded_array.isscalar
        assert loaded_scalar.format == 'isot'
        assert loaded_array.format == 'isot'

    def test_time_very_high_precision(self):
        """Test time with maximum precision"""
        t = Time('2023-01-01T12:30:45.123456789', format='isot', precision=9)

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        assert loaded.precision == 9
        assert loaded.format == 'isot'

    def test_time_zero_precision(self):
        """Test time with zero precision (no fractional seconds)"""
        t = Time('2023-01-01T12:30:45', format='isot', precision=0)

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        assert loaded.precision == 0
        assert '.' not in str(loaded)
