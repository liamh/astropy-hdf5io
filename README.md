# astropy-hdf5io

AstroPy serialization support for [fsc.hdf5-io](https://fsc-hdf5-io.readthedocs.io/).

## Installation

```bash
pip install astropy-hdf5io
```

## Supported Types

### Core Types
- ✅ `astropy.units.Quantity` - scalar and array values

### Coordinates
- ✅ `astropy.coordinates.Angle` - angular values
- ✅ `astropy.coordinates.Longitude` - longitude with wrap angle
- ✅ `astropy.coordinates.Latitude` - latitude values
- ✅ `astropy.coordinates.Distance` - distance/parallax
- ✅ `astropy.coordinates.EarthLocation` - observatory locations
- ✅ `astropy.coordinates.SkyCoord` - celestial coordinates

### Representations
- ✅ `astropy.coordinates.CartesianRepresentation` - Cartesian (x, y, z)
- ✅ `astropy.coordinates.SphericalRepresentation` - Spherical (lon, lat, distance)
- ✅ `astropy.coordinates.CylindricalRepresentation` - Cylindrical (rho, phi, z)
- ✅ `astropy.coordinates.PhysicsSphericalRepresentation` - Physics convention (phi, theta, r)

### Differentials (Velocities/Proper Motion)
- ✅ `astropy.coordinates.CartesianDifferential` - Cartesian velocities
- ✅ `astropy.coordinates.SphericalDifferential` - Spherical velocities
- ✅ `astropy.coordinates.SphericalCosLatDifferential` - Proper motion
- ✅ `astropy.coordinates.CylindricalDifferential` - Cylindrical velocities

### Time
- ✅ `astropy.time.Time` - various formats and scales

## Supported Types

### Tables and Time Series
- ✅ `astropy.table.Table` - general tabular data
- ✅ `astropy.table.QTable` - tables with Quantity columns
- ✅ `astropy.timeseries.TimeSeries` - time-indexed data with Quantities

## Usage

```
import astropy.units as u
from astropy.coordinates import SkyCoord
from fsc.hdf5_io import save, load
import astropy_hdf5io  # Registers serializers

# Save Quantities
distance = 1171 * u.Mpc
save(distance, 'distance.hdf5')
loaded_distance = load('distance.hdf5')

# Save SkyCoord
coord = SkyCoord(ra=10*u.degree, dec=20*u.degree)
save(coord, 'coordinate.hdf5')
loaded_coord = load('coordinate.hdf5')

# Works with nested structures
data = {
    'distance': 778 * u.kpc,
    'coordinate': SkyCoord(ra=83*u.degree, dec=22*u.degree),
    'velocities': [100, 200, 300] * u.km/u.s
}
save(data, 'galaxy.hdf5')
loaded_data = load('galaxy.hdf5')
```

## Usage Examples

```python
from astropy.coordinates import EarthLocation, CartesianRepresentation
import astropy.units as u
from fsc.hdf5_io import save, load

# Observatory location
keck = EarthLocation.from_geodetic(
    lon=-155.4783*u.degree,
    lat=19.8260*u.degree,
    height=4145*u.m
)
save(keck, 'keck.hdf5')

# Galactic position with velocity
position = CartesianRepresentation(
    x=8.5*u.kpc, y=0*u.kpc, z=0.02*u.kpc,
    differentials=CartesianDifferential(
        d_x=0*u.km/u.s, d_y=220*u.km/u.s, d_z=7*u.km/u.s
    )
)
save(position, 'solar_position.hdf5')

### Usage Example - Tables

```python
from astropy.table import QTable
from astropy.timeseries import TimeSeries
from astropy.time import Time
import astropy.units as u
from fsc.hdf5_io import save, load

# QTable with mixed column types
catalog = QTable()
catalog['name'] = ['M31', 'M87', 'NGC1275']
catalog['distance'] = [778, 53500, 70000] * u.kpc
catalog['ra'] = [10.68, 187.71, 49.95] * u.degree
catalog['dec'] = [41.27, 12.39, 41.51] * u.degree
catalog['flux'] = [1.5e-15, 3.2e-15, 2.1e-15] * u.erg / u.s / u.cm**2

save(catalog, 'galaxy_catalog.hdf5')
loaded_catalog = load('galaxy_catalog.hdf5')

# TimeSeries for light curves
times = Time(['2023-01-01T00:00:00',
              '2023-01-01T01:00:00',
              '2023-01-01T02:00:00'])

lightcurve = TimeSeries(time=times)
lightcurve['magnitude'] = [15.2, 15.1, 15.3]
lightcurve['flux'] = [1.2, 1.3, 1.1] * u.Jy

save(lightcurve, 'lightcurve.hdf5')
loaded_lc = load('lightcurve.hdf5')
```



# astropy-hdf5io

[![PyPI version](https://badge.fury.io/py/astropy-hdf5io.svg)](https://badge.fury.io/py/astropy-hdf5io)
[![Tests](https://github.com/liamh/astropy-hdf5io/workflows/Tests/badge.svg)](https://github.com/liamh/astropy-hdf5io/actions)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

HDF5 serialization support for AstroPy objects using `fsc.hdf5-io`.

## Features

- **Seamless integration** with `fsc.hdf5-io` for saving/loading AstroPy objects to HDF5
- **Comprehensive type support** including:
  - `Quantity` (with units, including logarithmic units)
  - `Time` (all formats and scales)
  - `SkyCoord` and coordinate frames
  - `Table`, `QTable`, and `TimeSeries`
  - Coordinate representations and differentials
  - `EarthLocation`, `Angle`, `Longitude`, `Latitude`, `Distance`
- **Metadata preservation** for tables and columns
- **Nested structures** support (tables in dicts/lists)
- **Masked columns** support

## Installation

```bash
pip install astropy-hdf5io
```

### Requirements

- Python ≥ 3.9
- astropy ≥ 5.0
- fsc.hdf5-io ≥ 1.0
- h5py ≥ 3.0
- numpy ≥ 1.20

## Quick Start

```python
import astropy.units as u
from astropy.table import QTable
from astropy.time import Time
from astropy.coordinates import SkyCoord
from fsc.hdf5_io import save, load

# Just import astropy_hdf5io to enable serialization
import astropy_hdf5io

# Create AstroPy objects
distance = 1171 * u.Mpc
time = Time('2023-01-01T00:00:00')
coord = SkyCoord(ra=10.68458*u.degree, dec=41.26917*u.degree, distance=distance)

# Save to HDF5
save(distance, 'distance.hdf5')
save(time, 'time.hdf5')
save(coord, 'coord.hdf5')

# Load from HDF5
loaded_distance = load('distance.hdf5')
loaded_time = load('time.hdf5')
loaded_coord = load('coord.hdf5')

print(loaded_distance)  # 1171.0 Mpc
print(loaded_time)      # 2023-01-01 00:00:00.000
print(loaded_coord)     # <SkyCoord ...>
```

## Examples

### Working with Quantities

```python
import astropy.units as u
from fsc.hdf5_io import save, load
import astropy_hdf5io

# Scalar and array quantities
distance = 42.0 * u.pc
velocities = [100, 200, 300] * u.km / u.s

save(distance, 'distance.hdf5')
save(velocities, 'velocities.hdf5')

loaded_distance = load('distance.hdf5')
loaded_velocities = load('velocities.hdf5')

# Complex units work too!
luminosity = 1e10 * u.solLum
flux = 1.2e-15 * u.erg / u.s / u.cm**2
```

### Working with Times

```python
from astropy.time import Time
from fsc.hdf5_io import save, load
import astropy_hdf5io

# Different time formats
t_iso = Time('2023-01-01T12:00:00')
t_jd = Time(2459945.5, format='jd')
t_mjd = Time(59945.0, format='mjd')

# Different time scales
t_utc = Time('2023-01-01', scale='utc')
t_tai = Time('2023-01-01', scale='tai')

save(t_iso, 'time.hdf5')
loaded = load('time.hdf5')
```

### Working with Coordinates

```python
from astropy.coordinates import SkyCoord
import astropy.units as u
from fsc.hdf5_io import save, load
import astropy_hdf5io

# 2D coordinates
coord_2d = SkyCoord(ra=10.68*u.degree, dec=41.27*u.degree, frame='icrs')

# 3D coordinates with distance
coord_3d = SkyCoord(ra=10.68*u.degree, dec=41.27*u.degree,
                     distance=770*u.kpc, frame='icrs')

# Arrays of coordinates
coords = SkyCoord(ra=[10, 20, 30]*u.degree,
                  dec=[40, 50, 60]*u.degree)

save(coord_3d, 'coord.hdf5')
loaded = load('coord.hdf5')
```

### Working with Tables

```python
from astropy.table import Table, QTable
import astropy.units as u
from fsc.hdf5_io import save, load
import astropy_hdf5io

# Regular Table
t = Table({
    'name': ['Star A', 'Star B', 'Star C'],
    'magnitude': [10.5, 12.3, 11.8],
    'distance': [100, 150, 120]
})

# QTable with Quantities
qt = QTable({
    'name': ['Galaxy 1', 'Galaxy 2'],
    'redshift': [0.1, 0.2],
    'distance': [500, 1000] * u.Mpc,
    'flux': [1.2e-15, 8.5e-16] * u.erg / u.s / u.cm**2
})

# Add metadata
qt.meta['telescope'] = 'HST'
qt.meta['observer'] = 'J. Smith'
qt['distance'].info.description = 'Luminosity distance'

save(qt, 'galaxies.hdf5')
loaded = load('galaxies.hdf5')

print(loaded.meta['telescope'])  # 'HST'
print(loaded['distance'].info.description)  # 'Luminosity distance'
```

### Working with TimeSeries

```python
from astropy.timeseries import TimeSeries
from astropy.time import Time
import astropy.units as u
from fsc.hdf5_io import save, load
import astropy_hdf5io

times = Time(['2023-01-01T00:00:00',
              '2023-01-01T01:00:00',
              '2023-01-01T02:00:00'])

ts = TimeSeries(time=times)
ts['flux'] = [100, 120, 110] * u.Jy
ts['temperature'] = [5800, 5850, 5820] * u.K

ts.meta['target'] = 'Variable Star XYZ'

save(ts, 'timeseries.hdf5')
loaded = load('timeseries.hdf5')
```

### Complex Nested Structures

```python
from astropy.table import QTable
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from fsc.hdf5_io import save, load
import astropy_hdf5io

# Complex nested data structure
observation_data = {
    'metadata': {
        'telescope': 'VLT',
        'observer': 'J. Smith',
        'date': Time('2023-01-01')
    },
    'targets': [
        SkyCoord(ra=10*u.degree, dec=40*u.degree, distance=1000*u.pc),
        SkyCoord(ra=20*u.degree, dec=50*u.degree, distance=2000*u.pc)
    ],
    'photometry': QTable({
        'time': Time(['2023-01-01', '2023-01-02']),
        'flux': [1.2, 1.5] * u.Jy,
        'magnitude': [18.5, 18.3]
    })
}

save(observation_data, 'observation.hdf5')
loaded = load('observation.hdf5')
```

## How It Works

`astropy-hdf5io` extends `fsc.hdf5-io` by:

1. **Monkey-patching** `to_hdf5()` methods onto AstroPy classes
2. **Registering deserializers** using the `@subscribe_hdf5` decorator
3. **Preserving metadata** including units, coordinate frames, and table information

Simply importing `astropy_hdf5io` automatically registers all serializers. After that, you can use `fsc.hdf5_io.save()` and `fsc.hdf5_io.load()` with AstroPy objects seamlessly.

## Supported Types

### Units and Quantities
- `astropy.units.Quantity` (including logarithmic units like magnitudes)

### Time
- `astropy.time.Time` (all formats: ISO, JD, MJD, etc.)

### Coordinates
- `astropy.coordinates.SkyCoord`
- `astropy.coordinates.Angle`
- `astropy.coordinates.Longitude`
- `astropy.coordinates.Latitude`
- `astropy.coordinates.Distance`
- `astropy.coordinates.EarthLocation`

### Representations
- `CartesianRepresentation`
- `SphericalRepresentation`
- `CylindricalRepresentation`
- `PhysicsSphericalRepresentation`

### Differentials
- `CartesianDifferential`
- `SphericalDifferential`
- `SphericalCosLatDifferential`
- `CylindricalDifferential`

### Tables
- `astropy.table.Table` (including masked columns)
- `astropy.table.QTable` (with Quantity columns)
- `astropy.timeseries.TimeSeries`

## Development

### Setup

```bash
git clone https://github.com/liamh/astropy-hdf5io.git
cd astropy-hdf5io
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/ -v
```

### Building Documentation

```bash
cd docs
make html
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run the test suite (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of [fsc.hdf5-io](https://github.com/Z2PackDev/fsc.hdf5_io)
- Designed for seamless integration with [AstroPy](https://www.astropy.org/)
- Inspired by the AstroPy community's need for efficient HDF5 storage

## Citation

If you use this package in your research, please cite:

```bibtex
@software{astropy_hdf5io,
  author = {{astropy-hdf5io contributors}},
  title = {astropy-hdf5io: HDF5 serialization for AstroPy},
  url = {https://github.com/liamh/astropy-hdf5io},
  version = {0.1.0},
  year = {2026}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/liamh/astropy-hdf5io/issues)
- **Discussions**: [GitHub Discussions](https://github.com/liamh/astropy-hdf5io/discussions)
- **Documentation**: [Read the Docs](https://astropy-hdf5io.readthedocs.io/)
