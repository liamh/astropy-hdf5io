"""Tests for recursive save/load functionality."""
import pytest
import tempfile
import os
import numpy as np
import astropy.units as u
from astropy.time import Time

from astropy_hdf5io import save_recursive, load_recursive, print_tree

# Optional munch import — used only in munch-specific tests
try:
    from munch import Munch, munchify
    HAS_MUNCH = True
except ImportError:
    HAS_MUNCH = False

requires_munch = pytest.mark.skipif(not HAS_MUNCH, reason="munch not installed")

class TestRecursiveSave:
    """Tests for save_recursive."""

    def test_simple_nested_structure(self):
        """Test saving a simple nested dict."""
        data = {
            'level1': {
                'value': 10.0 * u.m,
                'nested': {
                    'deep_value': 20.0 * u.s
                }
            }
        }

        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name

        try:
            saved = save_recursive(data, filename)

            assert 'level1/value' in saved
            assert 'level1/nested/deep_value' in saved
        finally:
            os.unlink(filename)

    @requires_munch
    def test_munch_structure(self):
        """Test saving a Munch structure."""
        data = Munch()
        data.init = Munch()
        data.init.value = 42.0 * u.km
        data.propn = Munch()
        data.propn.result = 100.0 * u.m

        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name

        try:
            saved = save_recursive(data, filename, 'study')

            # Paths are relative to base_path, so 'study/' is not included
            assert 'init/value' in saved
            assert 'propn/result' in saved
        finally:
            os.unlink(filename)

    @requires_munch
    def test_roundtrip_munch_structure(self):
        """Test saving and loading Munch structure."""
        original = Munch()
        original.data = Munch()
        original.data.distance = 100.0 * u.km
        original.data.time = Time('2025-01-01T00:00:00')
        original.results = Munch()
        original.results.value = 42.0 * u.m

        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name

        try:
            save_recursive(original, filename, 'study')
            loaded_dict = load_recursive(filename, 'study')
            loaded = munchify(loaded_dict)

            # When loading from 'study' base path, we get the structure
            # that was inside 'study', not nested under another 'study'
            assert hasattr(loaded, 'data')
            assert hasattr(loaded, 'results')
            assert hasattr(loaded.data, 'distance')
            assert loaded.data.distance == original.data.distance
        finally:
            os.unlink(filename)

    def test_skip_types(self):
        """Test skipping certain types."""
        data = {
            'value': 10.0 * u.m,
            'generator': (x for x in range(10)),
            'function': lambda x: x**2
        }

        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name

        try:
            import types
            saved = save_recursive(data, filename,
                                  skip_types=[types.GeneratorType, types.FunctionType])

            # Should only save the value
            assert 'value' in saved
            assert 'generator' not in saved
            assert 'function' not in saved
        finally:
            os.unlink(filename)


class TestRecursiveLoad:
    """Tests for load_recursive."""

    def test_load_nested_structure(self):
        """Test loading a nested structure."""
        data = {
            'level1': {
                'value': 10.0 * u.m,
                'level2': {
                    'deep': 20.0 * u.s
                }
            }
        }

        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name

        try:
            save_recursive(data, filename)
            loaded = load_recursive(filename)

            assert 'level1' in loaded
            assert 'value' in loaded['level1']
            assert 'level2' in loaded['level1']
            assert 'deep' in loaded['level1']['level2']
        finally:
            os.unlink(filename)

    def test_load_flat(self):
        """Test loading as flat dictionary."""
        data = {'a': {'b': 10.0 * u.m}}

        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name

        try:
            save_recursive(data, filename)
            loaded = load_recursive(filename, reconstruct_dicts=False)

            assert 'a/b' in loaded
        finally:
            os.unlink(filename)

    def test_load_subsection(self):
        """Test loading only a subsection."""
        data = {
            'section1': {'value': 10.0 * u.m},
            'section2': {'value': 20.0 * u.s}
        }

        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name

        try:
            save_recursive(data, filename)
            loaded = load_recursive(filename, 'section1')

            assert 'value' in loaded
            assert 'section2' not in loaded
        finally:
            os.unlink(filename)


class TestIntegration:
    """Integration tests."""

    @requires_munch
    def test_roundtrip_munch_structure(self):
        """Test saving and loading Munch structure."""
        original = Munch()
        original.data = Munch()
        original.data.distance = 100.0 * u.km
        original.data.time = Time('2025-01-01T00:00:00')
        original.results = Munch()
        original.results.value = 42.0 * u.m

        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name

        try:
            save_recursive(original, filename, 'study')
            loaded_dict = load_recursive(filename, 'study')
            loaded = munchify(loaded_dict)

            # Verify structure
            assert hasattr(loaded, 'data')
            assert hasattr(loaded.data, 'distance')
            assert loaded.data.distance == original.data.distance
        finally:
            os.unlink(filename)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
