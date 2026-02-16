# Contributing to astropy-hdf5io

Thank you for considering contributing to astropy-hdf5io!

## Development Setup

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/liamh/astropy-hdf5io.git
   cd astropy-hdf5io
   ```
3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and add tests

3. Run the test suite:
   ```bash
   pytest tests/ -v
   ```

4. Format your code:
   ```bash
   black src/ tests/
   ruff check src/ tests/
   ```

5. Commit your changes:
   ```bash
   git commit -m "Add feature: description"
   ```

6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Open a Pull Request on GitHub

## Testing

All new features must include tests. We aim for high test coverage.

```bash
# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=astropy_hdf5io --cov-report=html
```

## Code Style

We use:

- Black for code formatting
- Ruff for linting
- Type hints where appropriate

## Documentation

Update documentation when adding new features:

- Add docstrings to all public functions/classes
- Update README.md with usage examples
- Update CHANGELOG.md

## Adding Support for New AstroPy Types

If you want to add serialization support for a new AstroPy type:

1. Create a new file in `src/astropy_hdf5io/` (e.g., `_newtype.py`)

2. Implement serialization by monkey-patching a `to_hdf5` method:
   ```python
   def _newtype_to_hdf5(self, hdf5_handle):
       """Serialize NewType to HDF5"""
       hdf5_handle['type_tag'] = 'astropy_hdf5io.NewType'
       # Store data...

   NewType.to_hdf5 = _newtype_to_hdf5
   ```

3. Implement deserialization using `@subscribe_hdf5`:
   ```python
   from fsc.hdf5_io import subscribe_hdf5

   @subscribe_hdf5('astropy_hdf5io.NewType', check_on_load=False)
   class _NewTypeDeserializer:
       @classmethod
       def from_hdf5(cls, hdf5_handle):
           """Deserialize NewType from HDF5"""
           # Load data...
           return NewType(...)
   ```

4. Import your module in `src/astropy_hdf5io/__init__.py`

5. Add comprehensive tests in `tests/test_newtype.py`

6. Update README.md with examples

## Reporting Bugs

Please include:

- Python version
- astropy version
- fsc.hdf5-io version
- Minimal code to reproduce the issue
- Expected vs actual behavior
- Full error traceback

## Feature Requests

Open an issue describing:

- What AstroPy type/feature you'd like supported
- Your use case
- Example of how you'd like it to work

## Code Review Process

All submissions require review. We use GitHub pull requests for this purpose.

## Questions?

Feel free to open an issue for discussion before starting work on major features.

## Code of Conduct

Be respectful and constructive in all interactions. We're all here to make astronomy software better!