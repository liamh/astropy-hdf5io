"""Tests for HDF5 group utilities."""
import pytest
import tempfile
import os
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from astropy_hdf5io import (
    save_to_group,
    load_from_group,
    list_groups,
    print_tree,
    delete_group,
)


class TestSaveLoadGroup:
    """Tests for save_to_group and load_from_group."""

    def test_save_load_simple_group(self):
        """Test saving and loading from a simple group."""
        q = 42.0 * u.m
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name
        
        try:
            save_to_group(q, filename, 'data')
            loaded = load_from_group(filename, 'data')
            
            assert loaded == q
        finally:
            os.unlink(filename)

    def test_save_load_nested_hierarchy(self):
        """Test saving and loading from nested groups."""
        q = 100.0 * u.km
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name
        
        try:
            save_to_group(q, filename, 'experiment/run_001/results/distance')
            loaded = load_from_group(filename, 'experiment/run_001/results/distance')
            
            assert loaded == q
        finally:
            os.unlink(filename)

    def test_save_multiple_groups(self):
        """Test saving multiple objects to different groups."""
        q1 = 10.0 * u.m
        q2 = 20.0 * u.s
        q3 = 30.0 * u.kg
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name
        
        try:
            save_to_group(q1, filename, 'data/length')
            save_to_group(q2, filename, 'data/time')
            save_to_group(q3, filename, 'data/mass')
            
            loaded1 = load_from_group(filename, 'data/length')
            loaded2 = load_from_group(filename, 'data/time')
            loaded3 = load_from_group(filename, 'data/mass')
            
            assert loaded1 == q1
            assert loaded2 == q2
            assert loaded3 == q3
        finally:
            os.unlink(filename)

    def test_save_skycoord_to_group(self):
        """Test saving SkyCoord to a group."""
        coord = SkyCoord(ra=10*u.degree, dec=40*u.degree, distance=1000*u.pc)
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name
        
        try:
            save_to_group(coord, filename, 'observations/target1')
            loaded = load_from_group(filename, 'observations/target1')
            
            assert isinstance(loaded, SkyCoord)
            assert loaded.ra == coord.ra
            assert loaded.dec == coord.dec
        finally:
            os.unlink(filename)

    def test_save_time_to_group(self):
        """Test saving Time to a group."""
        time = Time('2023-01-01T00:00:00')
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name
        
        try:
            save_to_group(time, filename, 'metadata/epoch')
            loaded = load_from_group(filename, 'metadata/epoch')
            
            assert isinstance(loaded, Time)
            assert loaded == time
        finally:
            os.unlink(filename)

    def test_mode_append(self):
        """Test append mode preserves existing data."""
        q1 = 10.0 * u.m
        q2 = 20.0 * u.m
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name
        
        try:
            # Save first object
            save_to_group(q1, filename, 'data1', mode='w')
            
            # Save second object in append mode
            save_to_group(q2, filename, 'data2', mode='a')
            
            # Both should exist
            loaded1 = load_from_group(filename, 'data1')
            loaded2 = load_from_group(filename, 'data2')
            
            assert loaded1 == q1
            assert loaded2 == q2
        finally:
            os.unlink(filename)

    def test_mode_write_overwrites(self):
        """Test write mode overwrites file."""
        q1 = 10.0 * u.m
        q2 = 20.0 * u.m
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name
        
        try:
            # Save first object
            save_to_group(q1, filename, 'data1', mode='w')
            
            # Save second object in write mode (overwrites)
            save_to_group(q2, filename, 'data2', mode='w')
            
            # Only data2 should exist
            with pytest.raises(KeyError):
                load_from_group(filename, 'data1')
            
            loaded2 = load_from_group(filename, 'data2')
            assert loaded2 == q2
        finally:
            os.unlink(filename)

    def test_load_nonexistent_group(self):
        """Test loading from nonexistent group raises KeyError."""
        q = 10.0 * u.m
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name
        
        try:
            save_to_group(q, filename, 'data')
            
            with pytest.raises(KeyError):
                load_from_group(filename, 'nonexistent')
        finally:
            os.unlink(filename)


class TestListGroups:
    """Tests for list_groups."""

    def test_list_root_groups(self):
        """Test listing groups at root level."""
        q1 = 10.0 * u.m
        q2 = 20.0 * u.s
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name
        
        try:
            save_to_group(q1, filename, 'group1')
            save_to_group(q2, filename, 'group2')
            
            contents = list_groups(filename)
            
            assert set(contents['groups']) == {'group1', 'group2'}
        finally:
            os.unlink(filename)

    def test_list_nested_groups(self):
        """Test listing groups in nested hierarchy."""
        q1 = 10.0 * u.m
        q2 = 20.0 * u.s
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name
        
        try:
            save_to_group(q1, filename, 'experiment/run_001/data')
            save_to_group(q2, filename, 'experiment/run_002/data')
            
            contents = list_groups(filename, 'experiment')
            
            assert set(contents['groups']) == {'run_001', 'run_002'}
        finally:
            os.unlink(filename)


class TestPrintTree:
    """Tests for print_tree."""

    def test_print_tree_simple(self, capsys):
        """Test printing simple tree structure."""
        q = 10.0 * u.m
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name
        
        try:
            save_to_group(q, filename, 'data/value')
            
            print_tree(filename)
            captured = capsys.readouterr()
            
            assert 'data/' in captured.out
            assert 'value' in captured.out
        finally:
            os.unlink(filename)

    def test_print_tree_with_max_depth(self, capsys):
        """Test printing tree with depth limit."""
        q = 10.0 * u.m
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name
        
        try:
            save_to_group(q, filename, 'level1/level2/level3/data')
            
            print_tree(filename, max_depth=2)
            captured = capsys.readouterr()
            
            assert 'level1/' in captured.out
            assert 'level2/' in captured.out
            # level3 might not appear due to depth limit
        finally:
            os.unlink(filename)


class TestDeleteGroup:
    """Tests for delete_group."""

    def test_delete_existing_group(self):
        """Test deleting an existing group."""
        q1 = 10.0 * u.m
        q2 = 20.0 * u.m
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name
        
        try:
            save_to_group(q1, filename, 'keep')
            save_to_group(q2, filename, 'delete')
            
            delete_group(filename, 'delete')
            
            # 'keep' should still exist
            loaded = load_from_group(filename, 'keep')
            assert loaded == q1
            
            # 'delete' should not exist
            with pytest.raises(KeyError):
                load_from_group(filename, 'delete')
        finally:
            os.unlink(filename)

    def test_delete_nonexistent_group(self):
        """Test deleting nonexistent group raises KeyError."""
        q = 10.0 * u.m
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name
        
        try:
            save_to_group(q, filename, 'data')
            
            with pytest.raises(KeyError, match="not found"):
                delete_group(filename, 'nonexistent')
        finally:
            os.unlink(filename)


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_organize_astronomical_observations(self):
        """Test organizing multiple observations in hierarchy."""
        # Create observations
        target1 = SkyCoord(ra=10*u.degree, dec=40*u.degree, distance=1000*u.pc)
        target2 = SkyCoord(ra=20*u.degree, dec=50*u.degree, distance=2000*u.pc)
        obs_time = Time('2023-01-01T00:00:00')
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            filename = f.name
        
        try:
            # Save in organized hierarchy
            save_to_group(target1, filename, 'observations/2023/run_001/target')
            save_to_group(obs_time, filename, 'observations/2023/run_001/time')
            save_to_group(target2, filename, 'observations/2023/run_002/target')
            
            # Verify structure
            contents = list_groups(filename, 'observations/2023')
            assert 'run_001' in contents['groups']
            assert 'run_002' in contents['groups']
            
            # Load specific data
            loaded_target = load_from_group(filename, 'observations/2023/run_001/target')
            assert isinstance(loaded_target, SkyCoord)
        finally:
            os.unlink(filename)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])