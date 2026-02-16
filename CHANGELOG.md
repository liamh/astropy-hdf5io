# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-16

### Added
- Initial release
- Support for `astropy.units.Quantity` serialization
- Support for `astropy.time.Time` serialization
- Support for `astropy.coordinates.SkyCoord` serialization
- Support for coordinate representations and differentials
- Support for `astropy.table.Table` and `QTable`
- Support for `astropy.timeseries.TimeSeries`
- Comprehensive test suite with 62 tests
- Documentation and examples

### Features
- Automatic registration of serializers on import
- Metadata preservation for tables and columns
- Support for masked columns
- Support for nested structures (dicts, lists)
- Unicode string handling
- Logarithmic unit support (magnitudes, decibels)

[0.1.0]: https://github.com/liamh/astropy-hdf5io/releases/tag/v0.1.0
