"""Serialization support for astropy.coordinates types"""

import numpy as np
from fsc.hdf5_io import subscribe_hdf5, to_hdf5, from_hdf5
from astropy.coordinates import (
    EarthLocation, Angle, Longitude, Latitude, Distance,
    CartesianRepresentation, SphericalRepresentation,
    CylindricalRepresentation, PhysicsSphericalRepresentation,
    CartesianDifferential, SphericalDifferential,
    SphericalCosLatDifferential, CylindricalDifferential
)
import astropy.units as u


# Angle
def _angle_to_hdf5(self, hdf5_handle):
    """Serialize Angle to HDF5"""
    hdf5_handle['type_tag'] = 'astropy_hdf5io.Angle'
    to_hdf5(self.view(u.Quantity), hdf5_handle.create_group('quantity'))


Angle.to_hdf5 = _angle_to_hdf5


@subscribe_hdf5('astropy_hdf5io.Angle', check_on_load=False)
class _AngleDeserializer:
    @classmethod
    def from_hdf5(cls, hdf5_handle):
        quantity = from_hdf5(hdf5_handle['quantity'])
        return Angle(quantity)


# Longitude
def _longitude_to_hdf5(self, hdf5_handle):
    """Serialize Longitude to HDF5"""
    hdf5_handle['type_tag'] = 'astropy_hdf5io.Longitude'
    to_hdf5(self.view(u.Quantity), hdf5_handle.create_group('quantity'))
    hdf5_handle.attrs['wrap_angle'] = self.wrap_angle.to(u.degree).value


Longitude.to_hdf5 = _longitude_to_hdf5


@subscribe_hdf5('astropy_hdf5io.Longitude', check_on_load=False)
class _LongitudeDeserializer:
    @classmethod
    def from_hdf5(cls, hdf5_handle):
        quantity = from_hdf5(hdf5_handle['quantity'])
        wrap_angle = hdf5_handle.attrs['wrap_angle'] * u.degree
        return Longitude(quantity, wrap_angle=wrap_angle)


# Latitude
def _latitude_to_hdf5(self, hdf5_handle):
    """Serialize Latitude to HDF5"""
    hdf5_handle['type_tag'] = 'astropy_hdf5io.Latitude'
    to_hdf5(self.view(u.Quantity), hdf5_handle.create_group('quantity'))


Latitude.to_hdf5 = _latitude_to_hdf5


@subscribe_hdf5('astropy_hdf5io.Latitude', check_on_load=False)
class _LatitudeDeserializer:
    @classmethod
    def from_hdf5(cls, hdf5_handle):
        quantity = from_hdf5(hdf5_handle['quantity'])
        return Latitude(quantity)


# Distance
def _distance_to_hdf5(self, hdf5_handle):
    """Serialize Distance to HDF5"""
    hdf5_handle['type_tag'] = 'astropy_hdf5io.Distance'
    to_hdf5(self.view(u.Quantity), hdf5_handle.create_group('quantity'))

    if hasattr(self, '_parallax') and self._parallax is not None:
        to_hdf5(self._parallax, hdf5_handle.create_group('parallax'))
        hdf5_handle.attrs['has_parallax'] = True
    else:
        hdf5_handle.attrs['has_parallax'] = False


Distance.to_hdf5 = _distance_to_hdf5


@subscribe_hdf5('astropy_hdf5io.Distance', check_on_load=False)
class _DistanceDeserializer:
    @classmethod
    def from_hdf5(cls, hdf5_handle):
        quantity = from_hdf5(hdf5_handle['quantity'])

        if hdf5_handle.attrs['has_parallax']:
            parallax = from_hdf5(hdf5_handle['parallax'])
            return Distance(parallax=parallax)
        else:
            return Distance(quantity)


# EarthLocation
def _earthlocation_to_hdf5(self, hdf5_handle):
    """Serialize EarthLocation to HDF5"""
    hdf5_handle['type_tag'] = 'astropy_hdf5io.EarthLocation'
    to_hdf5(self.x, hdf5_handle.create_group('x'))
    to_hdf5(self.y, hdf5_handle.create_group('y'))
    to_hdf5(self.z, hdf5_handle.create_group('z'))


EarthLocation.to_hdf5 = _earthlocation_to_hdf5


@subscribe_hdf5('astropy_hdf5io.EarthLocation', check_on_load=False)
class _EarthLocationDeserializer:
    @classmethod
    def from_hdf5(cls, hdf5_handle):
        x = from_hdf5(hdf5_handle['x'])
        y = from_hdf5(hdf5_handle['y'])
        z = from_hdf5(hdf5_handle['z'])
        return EarthLocation(x=x, y=y, z=z)


# CartesianRepresentation
def _cartesian_repr_to_hdf5(self, hdf5_handle):
    """Serialize CartesianRepresentation to HDF5"""
    hdf5_handle['type_tag'] = 'astropy_hdf5io.CartesianRepresentation'
    to_hdf5(self.x, hdf5_handle.create_group('x'))
    to_hdf5(self.y, hdf5_handle.create_group('y'))
    to_hdf5(self.z, hdf5_handle.create_group('z'))

    if self.differentials:
        hdf5_handle.attrs['has_differentials'] = True
        diff_group = hdf5_handle.create_group('differentials')
        for key, diff in self.differentials.items():
            to_hdf5(diff, diff_group.create_group(key))
    else:
        hdf5_handle.attrs['has_differentials'] = False


CartesianRepresentation.to_hdf5 = _cartesian_repr_to_hdf5


@subscribe_hdf5('astropy_hdf5io.CartesianRepresentation', check_on_load=False)
class _CartesianRepresentationDeserializer:
    @classmethod
    def from_hdf5(cls, hdf5_handle):
        x = from_hdf5(hdf5_handle['x'])
        y = from_hdf5(hdf5_handle['y'])
        z = from_hdf5(hdf5_handle['z'])

        if hdf5_handle.attrs['has_differentials']:
            differentials = {}
            for key in hdf5_handle['differentials'].keys():
                differentials[key] = from_hdf5(hdf5_handle['differentials'][key])
            return CartesianRepresentation(x=x, y=y, z=z, differentials=differentials)
        else:
            return CartesianRepresentation(x=x, y=y, z=z)


# SphericalRepresentation
def _spherical_repr_to_hdf5(self, hdf5_handle):
    """Serialize SphericalRepresentation to HDF5"""
    hdf5_handle['type_tag'] = 'astropy_hdf5io.SphericalRepresentation'
    to_hdf5(self.lon, hdf5_handle.create_group('lon'))
    to_hdf5(self.lat, hdf5_handle.create_group('lat'))
    to_hdf5(self.distance, hdf5_handle.create_group('distance'))

    if self.differentials:
        hdf5_handle.attrs['has_differentials'] = True
        diff_group = hdf5_handle.create_group('differentials')
        for key, diff in self.differentials.items():
            to_hdf5(diff, diff_group.create_group(key))
    else:
        hdf5_handle.attrs['has_differentials'] = False


SphericalRepresentation.to_hdf5 = _spherical_repr_to_hdf5


@subscribe_hdf5('astropy_hdf5io.SphericalRepresentation', check_on_load=False)
class _SphericalRepresentationDeserializer:
    @classmethod
    def from_hdf5(cls, hdf5_handle):
        lon = from_hdf5(hdf5_handle['lon'])
        lat = from_hdf5(hdf5_handle['lat'])
        distance = from_hdf5(hdf5_handle['distance'])

        if hdf5_handle.attrs['has_differentials']:
            differentials = {}
            for key in hdf5_handle['differentials'].keys():
                differentials[key] = from_hdf5(hdf5_handle['differentials'][key])
            return SphericalRepresentation(lon=lon, lat=lat, distance=distance,
                                          differentials=differentials)
        else:
            return SphericalRepresentation(lon=lon, lat=lat, distance=distance)


# CylindricalRepresentation
def _cylindrical_repr_to_hdf5(self, hdf5_handle):
    """Serialize CylindricalRepresentation to HDF5"""
    hdf5_handle['type_tag'] = 'astropy_hdf5io.CylindricalRepresentation'
    to_hdf5(self.rho, hdf5_handle.create_group('rho'))
    to_hdf5(self.phi, hdf5_handle.create_group('phi'))
    to_hdf5(self.z, hdf5_handle.create_group('z'))

    if self.differentials:
        hdf5_handle.attrs['has_differentials'] = True
        diff_group = hdf5_handle.create_group('differentials')
        for key, diff in self.differentials.items():
            to_hdf5(diff, diff_group.create_group(key))
    else:
        hdf5_handle.attrs['has_differentials'] = False


CylindricalRepresentation.to_hdf5 = _cylindrical_repr_to_hdf5


@subscribe_hdf5('astropy_hdf5io.CylindricalRepresentation', check_on_load=False)
class _CylindricalRepresentationDeserializer:
    @classmethod
    def from_hdf5(cls, hdf5_handle):
        rho = from_hdf5(hdf5_handle['rho'])
        phi = from_hdf5(hdf5_handle['phi'])
        z = from_hdf5(hdf5_handle['z'])

        if hdf5_handle.attrs['has_differentials']:
            differentials = {}
            for key in hdf5_handle['differentials'].keys():
                differentials[key] = from_hdf5(hdf5_handle['differentials'][key])
            return CylindricalRepresentation(rho=rho, phi=phi, z=z,
                                           differentials=differentials)
        else:
            return CylindricalRepresentation(rho=rho, phi=phi, z=z)


# PhysicsSphericalRepresentation
def _physics_spherical_repr_to_hdf5(self, hdf5_handle):
    """Serialize PhysicsSphericalRepresentation to HDF5"""
    hdf5_handle['type_tag'] = 'astropy_hdf5io.PhysicsSphericalRepresentation'
    to_hdf5(self.phi, hdf5_handle.create_group('phi'))
    to_hdf5(self.theta, hdf5_handle.create_group('theta'))
    to_hdf5(self.r, hdf5_handle.create_group('r'))

    if self.differentials:
        hdf5_handle.attrs['has_differentials'] = True
        diff_group = hdf5_handle.create_group('differentials')
        for key, diff in self.differentials.items():
            to_hdf5(diff, diff_group.create_group(key))
    else:
        hdf5_handle.attrs['has_differentials'] = False


PhysicsSphericalRepresentation.to_hdf5 = _physics_spherical_repr_to_hdf5


@subscribe_hdf5('astropy_hdf5io.PhysicsSphericalRepresentation', check_on_load=False)
class _PhysicsSphericalRepresentationDeserializer:
    @classmethod
    def from_hdf5(cls, hdf5_handle):
        phi = from_hdf5(hdf5_handle['phi'])
        theta = from_hdf5(hdf5_handle['theta'])
        r = from_hdf5(hdf5_handle['r'])

        if hdf5_handle.attrs['has_differentials']:
            differentials = {}
            for key in hdf5_handle['differentials'].keys():
                differentials[key] = from_hdf5(hdf5_handle['differentials'][key])
            return PhysicsSphericalRepresentation(phi=phi, theta=theta, r=r,
                                                 differentials=differentials)
        else:
            return PhysicsSphericalRepresentation(phi=phi, theta=theta, r=r)


# CartesianDifferential
def _cartesian_diff_to_hdf5(self, hdf5_handle):
    """Serialize CartesianDifferential to HDF5"""
    hdf5_handle['type_tag'] = 'astropy_hdf5io.CartesianDifferential'
    to_hdf5(self.d_x, hdf5_handle.create_group('d_x'))
    to_hdf5(self.d_y, hdf5_handle.create_group('d_y'))
    to_hdf5(self.d_z, hdf5_handle.create_group('d_z'))


CartesianDifferential.to_hdf5 = _cartesian_diff_to_hdf5


@subscribe_hdf5('astropy_hdf5io.CartesianDifferential', check_on_load=False)
class _CartesianDifferentialDeserializer:
    @classmethod
    def from_hdf5(cls, hdf5_handle):
        d_x = from_hdf5(hdf5_handle['d_x'])
        d_y = from_hdf5(hdf5_handle['d_y'])
        d_z = from_hdf5(hdf5_handle['d_z'])
        return CartesianDifferential(d_x=d_x, d_y=d_y, d_z=d_z)


# SphericalDifferential
def _spherical_diff_to_hdf5(self, hdf5_handle):
    """Serialize SphericalDifferential to HDF5"""
    hdf5_handle['type_tag'] = 'astropy_hdf5io.SphericalDifferential'
    to_hdf5(self.d_lon, hdf5_handle.create_group('d_lon'))
    to_hdf5(self.d_lat, hdf5_handle.create_group('d_lat'))
    to_hdf5(self.d_distance, hdf5_handle.create_group('d_distance'))


SphericalDifferential.to_hdf5 = _spherical_diff_to_hdf5


@subscribe_hdf5('astropy_hdf5io.SphericalDifferential', check_on_load=False)
class _SphericalDifferentialDeserializer:
    @classmethod
    def from_hdf5(cls, hdf5_handle):
        d_lon = from_hdf5(hdf5_handle['d_lon'])
        d_lat = from_hdf5(hdf5_handle['d_lat'])
        d_distance = from_hdf5(hdf5_handle['d_distance'])
        return SphericalDifferential(d_lon=d_lon, d_lat=d_lat, d_distance=d_distance)


# SphericalCosLatDifferential
def _spherical_coslat_diff_to_hdf5(self, hdf5_handle):
    """Serialize SphericalCosLatDifferential to HDF5"""
    hdf5_handle['type_tag'] = 'astropy_hdf5io.SphericalCosLatDifferential'
    to_hdf5(self.d_lon_coslat, hdf5_handle.create_group('d_lon_coslat'))
    to_hdf5(self.d_lat, hdf5_handle.create_group('d_lat'))
    to_hdf5(self.d_distance, hdf5_handle.create_group('d_distance'))


SphericalCosLatDifferential.to_hdf5 = _spherical_coslat_diff_to_hdf5


@subscribe_hdf5('astropy_hdf5io.SphericalCosLatDifferential', check_on_load=False)
class _SphericalCosLatDifferentialDeserializer:
    @classmethod
    def from_hdf5(cls, hdf5_handle):
        d_lon_coslat = from_hdf5(hdf5_handle['d_lon_coslat'])
        d_lat = from_hdf5(hdf5_handle['d_lat'])
        d_distance = from_hdf5(hdf5_handle['d_distance'])
        return SphericalCosLatDifferential(d_lon_coslat=d_lon_coslat,
                                          d_lat=d_lat, d_distance=d_distance)


# CylindricalDifferential
def _cylindrical_diff_to_hdf5(self, hdf5_handle):
    """Serialize CylindricalDifferential to HDF5"""
    hdf5_handle['type_tag'] = 'astropy_hdf5io.CylindricalDifferential'
    to_hdf5(self.d_rho, hdf5_handle.create_group('d_rho'))
    to_hdf5(self.d_phi, hdf5_handle.create_group('d_phi'))
    to_hdf5(self.d_z, hdf5_handle.create_group('d_z'))


CylindricalDifferential.to_hdf5 = _cylindrical_diff_to_hdf5


@subscribe_hdf5('astropy_hdf5io.CylindricalDifferential', check_on_load=False)
class _CylindricalDifferentialDeserializer:
    @classmethod
    def from_hdf5(cls, hdf5_handle):
        d_rho = from_hdf5(hdf5_handle['d_rho'])
        d_phi = from_hdf5(hdf5_handle['d_phi'])
        d_z = from_hdf5(hdf5_handle['d_z'])
        return CylindricalDifferential(d_rho=d_rho, d_phi=d_phi, d_z=d_z)
