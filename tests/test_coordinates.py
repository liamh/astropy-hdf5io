"""Tests for astropy.coordinates serialization"""

import pytest
import numpy as np
import astropy.units as u
from astropy.coordinates import (
    EarthLocation, Angle, Longitude, Latitude, Distance,
    CartesianRepresentation, SphericalRepresentation,
    CylindricalRepresentation, PhysicsSphericalRepresentation,
    CartesianDifferential, SphericalDifferential,
    SphericalCosLatDifferential, CylindricalDifferential
)
from fsc.hdf5_io import save, load
from tempfile import NamedTemporaryFile

import astropy_hdf5io


class TestAngle:
    """Tests for Angle serialization"""

    def test_scalar_angle(self):
        """Test scalar Angle"""
        angle = Angle(45, unit=u.degree)

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(angle, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, Angle)
        assert np.isclose(loaded.degree, angle.degree)
        assert loaded.unit == angle.unit

    def test_array_angle(self):
        """Test array Angle"""
        angles = Angle([10, 20, 30], unit=u.degree)

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(angles, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, Angle)
        assert np.allclose(loaded.degree, angles.degree)

    def test_angle_in_radians(self):
        """Test Angle with different units"""
        angle = Angle(np.pi / 4, unit=u.radian)

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(angle, f.name)
            loaded = load(f.name)

        assert np.isclose(loaded.radian, angle.radian)

    def test_angle_hms(self):
        """Test Angle from hour:minute:second"""
        angle = Angle('1h30m45s')

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(angle, f.name)
            loaded = load(f.name)

        assert np.isclose(loaded.degree, angle.degree)


class TestLongitude:
    """Tests for Longitude serialization"""

    def test_longitude_default_wrap(self):
        """Test Longitude with default wrap angle"""
        lon = Longitude(350 * u.degree)

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(lon, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, Longitude)
        assert np.isclose(loaded.degree, lon.degree)
        assert loaded.wrap_angle == lon.wrap_angle

    def test_longitude_custom_wrap(self):
        """Test Longitude with custom wrap angle"""
        lon = Longitude(350 * u.degree, wrap_angle=180*u.degree)

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(lon, f.name)
            loaded = load(f.name)

        assert loaded.wrap_angle == lon.wrap_angle
        assert np.isclose(loaded.degree, lon.degree)

    def test_longitude_array(self):
        """Test array Longitude"""
        lon = Longitude([10, 350, 180] * u.degree)

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(lon, f.name)
            loaded = load(f.name)

        assert np.allclose(loaded.degree, lon.degree)


class TestLatitude:
    """Tests for Latitude serialization"""

    def test_latitude(self):
        """Test Latitude"""
        lat = Latitude(45 * u.degree)

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(lat, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, Latitude)
        assert np.isclose(loaded.degree, lat.degree)

    def test_latitude_array(self):
        """Test array Latitude"""
        lat = Latitude([-90, 0, 45, 90] * u.degree)

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(lat, f.name)
            loaded = load(f.name)

        assert np.allclose(loaded.degree, lat.degree)

    def test_latitude_dms(self):
        """Test Latitude from degree:minute:second"""
        lat = Latitude('45d30m15s')

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(lat, f.name)
            loaded = load(f.name)

        assert np.isclose(loaded.degree, lat.degree)


class TestDistance:
    """Tests for Distance serialization"""

    def test_distance(self):
        """Test Distance"""
        dist = Distance(100 * u.pc)

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(dist, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, Distance)
        assert np.isclose(loaded.pc, dist.pc)

    def test_distance_array(self):
        """Test array Distance"""
        dist = Distance([10, 100, 1000] * u.pc)

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(dist, f.name)
            loaded = load(f.name)

        assert np.allclose(loaded.pc, dist.pc)

    def test_distance_from_parallax(self):
        """Test Distance created from parallax"""
        dist = Distance(parallax=10 * u.mas)

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(dist, f.name)
            loaded = load(f.name)

        # Both should have same distance value
        assert np.isclose(loaded.pc, dist.pc)


class TestEarthLocation:
    """Tests for EarthLocation serialization"""

    def test_earth_location_from_geocentric(self):
        """Test EarthLocation from geocentric coordinates"""
        loc = EarthLocation(x=1000*u.km, y=2000*u.km, z=3000*u.km)

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(loc, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, EarthLocation)
        assert np.isclose(loaded.x.value, loc.x.value)
        assert np.isclose(loaded.y.value, loc.y.value)
        assert np.isclose(loaded.z.value, loc.z.value)

    def test_earth_location_from_geodetic(self):
        """Test EarthLocation from geodetic coordinates"""
        # Green Observatory
        loc = EarthLocation.from_geodetic(
            lon=-111.5967*u.degree,
            lat=31.9583*u.degree,
            height=2096*u.m
        )

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(loc, f.name)
            loaded = load(f.name)

        # Check that geocentric coordinates match
        assert np.isclose(loaded.x.value, loc.x.value)
        assert np.isclose(loaded.y.value, loc.y.value)
        assert np.isclose(loaded.z.value, loc.z.value)

        # Check that we can recover geodetic coordinates
        assert np.isclose(loaded.lon.degree, loc.lon.degree)
        assert np.isclose(loaded.lat.degree, loc.lat.degree)
        assert np.isclose(loaded.height.value, loc.height.value)

    def test_earth_location_of_site(self):
        """Test EarthLocation from known observatory"""
        loc = EarthLocation.of_site('greenwich')

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(loc, f.name)
            loaded = load(f.name)

        assert np.isclose(loaded.lat.degree, loc.lat.degree)
        assert np.isclose(loaded.lon.degree, loc.lon.degree)


class TestCartesianRepresentation:
    """Tests for CartesianRepresentation serialization"""

    def test_cartesian_representation(self):
        """Test CartesianRepresentation"""
        rep = CartesianRepresentation(x=1*u.kpc, y=2*u.kpc, z=3*u.kpc)

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(rep, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, CartesianRepresentation)
        assert np.isclose(loaded.x.value, rep.x.value)
        assert np.isclose(loaded.y.value, rep.y.value)
        assert np.isclose(loaded.z.value, rep.z.value)

    def test_cartesian_with_differential(self):
        """Test CartesianRepresentation with velocity"""
        rep = CartesianRepresentation(
            x=1*u.kpc, y=2*u.kpc, z=3*u.kpc,
            differentials=CartesianDifferential(
                d_x=10*u.km/u.s, d_y=20*u.km/u.s, d_z=30*u.km/u.s
            )
        )

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(rep, f.name)
            loaded = load(f.name)

        assert loaded.differentials
        vel = loaded.differentials['s']
        orig_vel = rep.differentials['s']
        assert np.isclose(vel.d_x.value, orig_vel.d_x.value)
        assert np.isclose(vel.d_y.value, orig_vel.d_y.value)
        assert np.isclose(vel.d_z.value, orig_vel.d_z.value)

    def test_cartesian_array(self):
        """Test array CartesianRepresentation"""
        rep = CartesianRepresentation(
            x=[1, 2, 3]*u.kpc,
            y=[4, 5, 6]*u.kpc,
            z=[7, 8, 9]*u.kpc
        )

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(rep, f.name)
            loaded = load(f.name)

        assert np.allclose(loaded.x.value, rep.x.value)
        assert np.allclose(loaded.y.value, rep.y.value)
        assert np.allclose(loaded.z.value, rep.z.value)


class TestSphericalRepresentation:
    """Tests for SphericalRepresentation serialization"""

    def test_spherical_representation(self):
        """Test SphericalRepresentation"""
        rep = SphericalRepresentation(
            lon=10*u.degree,
            lat=20*u.degree,
            distance=100*u.pc
        )

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(rep, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, SphericalRepresentation)
        assert np.isclose(loaded.lon.degree, rep.lon.degree)
        assert np.isclose(loaded.lat.degree, rep.lat.degree)
        assert np.isclose(loaded.distance.value, rep.distance.value)

    def test_spherical_with_differential(self):
        """Test SphericalRepresentation with proper motion"""
        rep = SphericalRepresentation(
            lon=10*u.degree,
            lat=20*u.degree,
            distance=100*u.pc,
            differentials=SphericalDifferential(
                d_lon=0.1*u.mas/u.yr,
                d_lat=0.2*u.mas/u.yr,
                d_distance=10*u.km/u.s
            )
        )

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(rep, f.name)
            loaded = load(f.name)

        assert loaded.differentials
        pm = loaded.differentials['s']
        orig_pm = rep.differentials['s']
        assert np.isclose(pm.d_lon.value, orig_pm.d_lon.value)
        assert np.isclose(pm.d_lat.value, orig_pm.d_lat.value)


class TestCylindricalRepresentation:
    """Tests for CylindricalRepresentation serialization"""

    def test_cylindrical_representation(self):
        """Test CylindricalRepresentation"""
        rep = CylindricalRepresentation(
            rho=1*u.kpc,
            phi=45*u.degree,
            z=0.5*u.kpc
        )

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(rep, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, CylindricalRepresentation)
        assert np.isclose(loaded.rho.value, rep.rho.value)
        assert np.isclose(loaded.phi.degree, rep.phi.degree)
        assert np.isclose(loaded.z.value, rep.z.value)


class TestPhysicsSphericalRepresentation:
    """Tests for PhysicsSphericalRepresentation serialization"""

    def test_physics_spherical_representation(self):
        """Test PhysicsSphericalRepresentation"""
        rep = PhysicsSphericalRepresentation(
            phi=45*u.degree,
            theta=60*u.degree,
            r=100*u.pc
        )

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(rep, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, PhysicsSphericalRepresentation)
        assert np.isclose(loaded.phi.degree, rep.phi.degree)
        assert np.isclose(loaded.theta.degree, rep.theta.degree)
        assert np.isclose(loaded.r.value, rep.r.value)


class TestDifferentials:
    """Tests for differential serialization"""

    def test_cartesian_differential(self):
        """Test CartesianDifferential"""
        diff = CartesianDifferential(
            d_x=10*u.km/u.s,
            d_y=20*u.km/u.s,
            d_z=30*u.km/u.s
        )

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(diff, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, CartesianDifferential)
        assert np.isclose(loaded.d_x.value, diff.d_x.value)
        assert np.isclose(loaded.d_y.value, diff.d_y.value)
        assert np.isclose(loaded.d_z.value, diff.d_z.value)

    def test_spherical_differential(self):
        """Test SphericalDifferential"""
        diff = SphericalDifferential(
            d_lon=0.1*u.mas/u.yr,
            d_lat=0.2*u.mas/u.yr,
            d_distance=10*u.km/u.s
        )

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(diff, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, SphericalDifferential)
        assert np.isclose(loaded.d_lon.value, diff.d_lon.value)
        assert np.isclose(loaded.d_lat.value, diff.d_lat.value)
        assert np.isclose(loaded.d_distance.value, diff.d_distance.value)

    def test_spherical_coslat_differential(self):
        """Test SphericalCosLatDifferential (proper motion)"""
        diff = SphericalCosLatDifferential(
            d_lon_coslat=5*u.mas/u.yr,
            d_lat=3*u.mas/u.yr,
            d_distance=10*u.km/u.s
        )

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(diff, f.name)
            loaded = load(f.name)

        assert isinstance(loaded, SphericalCosLatDifferential)
        assert np.isclose(loaded.d_lon_coslat.value, diff.d_lon_coslat.value)
        assert np.isclose(loaded.d_lat.value, diff.d_lat.value)


class TestNestedStructures:
    """Tests for coordinates in nested structures"""

    def test_coordinates_in_dict(self):
        """Test various coordinate types in dictionary"""
        data = {
            'angle': Angle(45*u.degree),
            'location': EarthLocation.of_site('greenwich'),
            'distance': Distance(100*u.pc),
            'position': CartesianRepresentation(1*u.kpc, 2*u.kpc, 3*u.kpc)
        }

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(data, f.name)
            loaded = load(f.name)

        assert isinstance(loaded['angle'], Angle)
        assert isinstance(loaded['location'], EarthLocation)
        assert isinstance(loaded['distance'], Distance)
        assert isinstance(loaded['position'], CartesianRepresentation)

    def test_observatory_data(self):
        """Test realistic observatory data structure"""
        data = {
            'name': 'Keck Observatory',
            'location': EarthLocation.from_geodetic(
                lon=-155.4783*u.degree,
                lat=19.8260*u.degree,
                height=4145*u.m
            ),
            'target_coords': SphericalRepresentation(
                lon=83.63*u.degree,
                lat=22.01*u.degree,
                distance=414*u.pc
            ),
            'altitude': Angle(45*u.degree),
            'azimuth': Angle(180*u.degree)
        }

        with NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            save(data, f.name)
            loaded = load(f.name)

        assert loaded['name'] == data['name']
        assert np.isclose(loaded['location'].lat.degree,
                         data['location'].lat.degree)
        assert np.isclose(loaded['target_coords'].distance.value,
                         data['target_coords'].distance.value)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
