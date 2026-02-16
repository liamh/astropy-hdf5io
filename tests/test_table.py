"""Tests for astropy.table and astropy.timeseries serialization"""

import pytest
import numpy as np
import astropy.units as u
from astropy.table import Table, QTable, Column, MaskedColumn
from astropy.timeseries import TimeSeries
from astropy.time import Time
from fsc.hdf5_io import save, load
from tempfile import NamedTemporaryFile

import astropy_hdf5io


class TestTable:
    """Tests for Table serialization"""

    def test_simple_table(self):
        """Test basic Table with numeric columns"""
        t = Table()
        t['name'] = ['Alice', 'Bob', 'Charlie']
        t['age'] = [25, 30, 35]
        t['height'] = [1.65, 1.80, 1.75]

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, Table)
        assert len(loaded) == len(t)
        assert loaded.colnames == t.colnames
        assert np.array_equal(loaded['name'], t['name'])
        assert np.array_equal(loaded['age'], t['age'])
        assert np.allclose(loaded['height'], t['height'])

    def test_table_with_metadata(self):
        """Test Table with metadata"""
        t = Table()
        t['x'] = [1, 2, 3]
        t['y'] = [4, 5, 6]

        # Add table metadata
        t.meta['instrument'] = 'HST'
        t.meta['exposure'] = 100.0
        t.meta['filter'] = 'F814W'

        # Add column metadata
        t['x'].description = 'X coordinate'
        t['x'].format = '.2f'
        t['x'].meta = {'calibrated': True}

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        assert loaded.meta['instrument'] == t.meta['instrument']
        assert loaded.meta['exposure'] == t.meta['exposure']
        assert loaded['x'].description == t['x'].description
        assert loaded['x'].format == t['x'].format

    def test_masked_table(self):
        """Test Table with masked values"""
        t = Table()
        t['data'] = MaskedColumn([1, 2, 3, 4, 5], mask=[0, 0, 1, 0, 1])

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        assert isinstance(loaded['data'], MaskedColumn)
        assert np.array_equal(loaded['data'].mask, t['data'].mask)
        assert np.array_equal(loaded['data'].data, t['data'].data)

    def test_empty_table(self):
        """Test empty Table"""
        t = Table(names=['a', 'b', 'c'], dtype=[int, float, str])

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        assert len(loaded) == 0
        assert loaded.colnames == t.colnames

    def test_table_with_units(self):
        """Test Table with unit information in columns"""
        t = Table()
        t['distance'] = Column([1, 2, 3], unit=u.km)
        t['time'] = Column([10, 20, 30], unit=u.s)

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        assert loaded['distance'].unit == u.km
        assert loaded['time'].unit == u.s


class TestQTable:
    """Tests for QTable serialization"""

    def test_simple_qtable(self):
        """Test QTable with Quantity columns"""
        t = QTable()
        t['distance'] = [1, 2, 3] * u.Mpc
        t['velocity'] = [100, 200, 300] * u.km / u.s
        t['name'] = ['NGC1', 'NGC2', 'NGC3']

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, QTable)
        assert len(loaded) == len(t)
        assert loaded.colnames == t.colnames

        # Check Quantity columns
        assert isinstance(loaded['distance'], u.Quantity)
        assert np.allclose(loaded['distance'].value, t['distance'].value)
        assert loaded['distance'].unit == t['distance'].unit

        assert isinstance(loaded['velocity'], u.Quantity)
        assert np.allclose(loaded['velocity'].value, t['velocity'].value)
        assert loaded['velocity'].unit == t['velocity'].unit

        # Check string column
        assert np.array_equal(loaded['name'], t['name'])

    def test_qtable_mixed_columns(self):
        """Test QTable with mix of Quantity and regular columns"""
        t = QTable()
        t['flux'] = [1.5, 2.3, 3.1] * u.Jy
        t['ra'] = [10.5, 20.3, 30.1] * u.degree
        t['dec'] = [45.2, 50.8, 55.3] * u.degree
        t['id'] = [101, 102, 103]
        t['detected'] = [True, True, False]

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, QTable)
        assert isinstance(loaded['flux'], u.Quantity)
        assert isinstance(loaded['ra'], u.Quantity)
        assert isinstance(loaded['dec'], u.Quantity)
        assert np.array_equal(loaded['id'], t['id'])
        assert np.array_equal(loaded['detected'], t['detected'])

    def test_qtable_with_metadata(self):
        """Test QTable with metadata"""
        t = QTable()
        t['luminosity'] = [1e10, 2e10, 3e10] * u.solLum

        t.meta['telescope'] = 'VLT'
        t.meta['date'] = '2023-01-15'
        t.meta['coordinates'] = {'ra': 180.0, 'dec': 45.0}

        t['luminosity'].info.description = 'Stellar luminosity'
        t['luminosity'].info.meta = {'method': 'photometric'}

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        assert loaded.meta['telescope'] == t.meta['telescope']
        assert loaded.meta['date'] == t.meta['date']
        assert loaded['luminosity'].info.description == t['luminosity'].info.description
        assert loaded['luminosity'].info.meta == t['luminosity'].info.meta

    def test_qtable_complex_units(self):
        """Test QTable with complex unit combinations"""
        t = QTable()
        t['energy'] = [1, 2, 3] * u.erg
        t['power'] = [100, 200, 300] * u.W
        t['area'] = [10, 20, 30] * u.m**2
        t['density'] = [1.5, 2.5, 3.5] * u.g / u.cm**3
        t['acceleration'] = [9.8, 10.2, 9.5] * u.m / u.s**2

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        for colname in t.colnames:
            assert loaded[colname].unit == t[colname].unit
            assert np.allclose(loaded[colname].value, t[colname].value)

    def test_qtable_array_quantities(self):
        """Test QTable with array-valued Quantity columns"""
        t = QTable()
        t['position'] = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ] * u.kpc

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(t, f.name)
            loaded = load(f.name)

        assert isinstance(loaded['position'], u.Quantity)
        assert loaded['position'].shape == t['position'].shape
        assert np.allclose(loaded['position'].value, t['position'].value)


class TestTimeSeries:
    """Tests for TimeSeries serialization"""

    def test_simple_timeseries(self):
        """Test basic TimeSeries"""
        times = Time(['2023-01-01T00:00:00',
                      '2023-01-01T01:00:00',
                      '2023-01-01T02:00:00'])

        ts = TimeSeries(time=times)
        ts['flux'] = [1.5, 2.3, 1.8] * u.Jy
        ts['magnitude'] = [15.2, 14.8, 15.5]

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(ts, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, TimeSeries)
        assert len(loaded) == len(ts)

        # Check time column
        assert np.allclose(loaded.time.jd, ts.time.jd)

        # Check data columns
        assert isinstance(loaded['flux'], u.Quantity)
        assert np.allclose(loaded['flux'].value, ts['flux'].value)
        assert loaded['flux'].unit == ts['flux'].unit

        assert np.allclose(loaded['magnitude'], ts['magnitude'])

    def test_timeseries_with_metadata(self):
        """Test TimeSeries with metadata"""
        times = Time(['2023-01-01T00:00:00',
                      '2023-01-01T01:00:00',
                      '2023-01-01T02:00:00'])

        ts = TimeSeries(time=times)
        ts['brightness'] = [100, 120, 110] * u.ABmag

        ts.meta['target'] = 'V838 Mon'
        ts.meta['telescope'] = 'Kepler'
        ts.meta['filter'] = 'V'

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(ts, f.name)
            loaded = load(f.name)

        assert loaded.meta['target'] == ts.meta['target']
        assert loaded.meta['telescope'] == ts.meta['telescope']
        assert loaded.meta['filter'] == ts.meta['filter']

    def test_timeseries_multiband(self):
        """Test TimeSeries with multiple bands"""
        times = Time(['2023-01-01T00:00:00',
                      '2023-01-01T01:00:00',
                      '2023-01-01T02:00:00',
                      '2023-01-01T03:00:00'])

        ts = TimeSeries(time=times)
        ts['flux_g'] = [1.2, 1.5, 1.3, 1.4] * u.Jy
        ts['flux_r'] = [2.1, 2.3, 2.2, 2.4] * u.Jy
        ts['flux_i'] = [3.0, 3.2, 3.1, 3.3] * u.Jy
        ts['seeing'] = [1.2, 1.5, 1.3, 1.1] * u.arcsec
        ts['airmass'] = [1.1, 1.2, 1.3, 1.4]

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(ts, f.name)
            loaded = load(f.name)

        assert len(loaded) == len(ts)
        for colname in ['flux_g', 'flux_r', 'flux_i', 'seeing']:
            assert isinstance(loaded[colname], u.Quantity)
            assert loaded[colname].unit == ts[colname].unit
            assert np.allclose(loaded[colname].value, ts[colname].value)

    def test_timeseries_from_qtable(self):
        """Test TimeSeries created from QTable"""
        # Create a QTable first
        qt = QTable()
        qt['time'] = Time(['2023-01-01T00:00:00',
                           '2023-01-01T01:00:00',
                           '2023-01-01T02:00:00'])
        qt['velocity'] = [10, 20, 15] * u.km / u.s
        qt['temperature'] = [300, 310, 305] * u.K

        # Convert to TimeSeries
        ts = TimeSeries(qt)

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(ts, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, TimeSeries)
        assert np.allclose(loaded.time.jd, ts.time.jd)
        assert np.allclose(loaded['velocity'].value, ts['velocity'].value)
        assert np.allclose(loaded['temperature'].value, ts['temperature'].value)

    @pytest.mark.xfail(reason="AstroPy does not support empty Time arrays")
    def test_timeseries_empty(self):
        """Test empty TimeSeries"""
        times = Time([])
        ts = TimeSeries(time=times)

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(ts, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, TimeSeries)
        assert len(loaded) == 0


class TestNestedTableStructures:
    """Tests for tables in nested structures"""

    def test_tables_in_dict(self):
        """Test multiple tables in dictionary"""
        data = {
            'observations': QTable({
                'time': Time(['2023-01-01', '2023-01-02']),
                'flux': [1.2, 1.5] * u.Jy
            }),
            'calibration': Table({
                'wavelength': [500, 600, 700],
                'response': [0.95, 0.98, 0.96]
            }),
            'metadata': {
                'telescope': 'VLT',
                'observer': 'J. Smith'
            }
        }

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(data, f.name)
            loaded = load(f.name)

        assert isinstance(loaded['observations'], QTable)
        assert isinstance(loaded['calibration'], Table)
        assert loaded['metadata']['telescope'] == data['metadata']['telescope']

    def test_table_in_list(self):
        """Test list of tables"""
        tables = [
            QTable({'flux': [1, 2] * u.Jy}),
            QTable({'flux': [3, 4] * u.Jy}),
            QTable({'flux': [5, 6] * u.Jy})
        ]

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(tables, f.name)
            loaded = load(f.name)

        assert len(loaded) == len(tables)
        for orig, load_tbl in zip(tables, loaded):
            assert isinstance(load_tbl, QTable)
            assert np.allclose(load_tbl['flux'].value, orig['flux'].value)


class TestRealWorldExamples:
    """Tests with realistic astronomical data"""

    def test_photometry_table(self):
        """Test realistic photometry table"""
        phot = QTable()
        phot['id'] = [1, 2, 3, 4, 5]
        phot['ra'] = [10.5, 20.3, 30.1, 40.8, 50.2] * u.degree
        phot['dec'] = [45.2, 50.8, 55.3, 60.1, 65.5] * u.degree
        phot['g_mag'] = [18.5, 19.2, 17.8, 20.1, 18.9]
        phot['r_mag'] = [17.8, 18.5, 17.1, 19.4, 18.2]
        phot['flux_g'] = [1.2e-15, 8.5e-16, 1.8e-15, 5.2e-16, 1.1e-15] * u.erg / u.s / u.cm**2
        phot['flux_r'] = [2.1e-15, 1.5e-15, 3.2e-15, 9.8e-16, 1.9e-15] * u.erg / u.s / u.cm**2

        phot.meta['telescope'] = 'LSST'
        phot.meta['date'] = '2023-10-15'
        phot.meta['exposure_time'] = 30.0
        phot.meta['seeing'] = 0.8

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(phot, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, QTable)
        assert len(loaded) == len(phot)
        assert loaded.colnames == phot.colnames

        # Check Quantity columns preserved
        for col in ['ra', 'dec', 'flux_g', 'flux_r']:
            assert isinstance(loaded[col], u.Quantity)
            assert loaded[col].unit == phot[col].unit

    def test_lightcurve_timeseries(self):
        """Test realistic light curve time series"""
        # Simulate a light curve
        times = Time('2023-01-01') + np.linspace(0, 100, 50) * u.day

        # Variable star with periodic variation
        phase = 2 * np.pi * np.linspace(0, 10, 50)
        flux = (1.0 + 0.2 * np.sin(phase)) * u.Jy
        flux_err = np.random.uniform(0.01, 0.03, 50) * u.Jy

        lc = TimeSeries(time=times)
        lc['flux'] = flux
        lc['flux_err'] = flux_err
        lc['filter'] = ['V'] * 50
        lc['airmass'] = np.random.uniform(1.0, 1.5, 50)

        lc.meta['target'] = 'RR Lyrae'
        lc.meta['period'] = 0.567  # days
        lc.meta['observatory'] = 'Palomar'

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(lc, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, TimeSeries)
        assert len(loaded) == len(lc)
        assert loaded.meta['target'] == lc.meta['target']
        assert np.allclose(loaded.time.jd, lc.time.jd)
        assert np.allclose(loaded['flux'].value, lc['flux'].value)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
