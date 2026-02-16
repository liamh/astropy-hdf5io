"""Serialization support for astropy.coordinates.SkyCoord"""

from fsc.hdf5_io import subscribe_hdf5, to_hdf5, from_hdf5
from astropy.coordinates import SkyCoord
import astropy.units as u


def _skycoord_to_hdf5(self, hdf5_handle):
    """Serialize SkyCoord to HDF5"""
    hdf5_handle['type_tag'] = 'astropy_hdf5io.SkyCoord'

    icrs = self.icrs

    # Store RA and Dec
    to_hdf5(icrs.ra, hdf5_handle.create_group('ra'))
    to_hdf5(icrs.dec, hdf5_handle.create_group('dec'))

    # Store distance if available
    if icrs.distance is not None and not icrs.distance.unit.is_equivalent(u.dimensionless_unscaled):
        to_hdf5(icrs.distance, hdf5_handle.create_group('distance'))
        hdf5_handle.attrs['has_distance'] = True
    else:
        hdf5_handle.attrs['has_distance'] = False

    hdf5_handle.attrs['frame'] = self.frame.name


SkyCoord.to_hdf5 = _skycoord_to_hdf5


@subscribe_hdf5('astropy_hdf5io.SkyCoord', check_on_load=False)
class _SkyCoordDeserializer:
    """Deserializer for SkyCoord"""

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        """Deserialize SkyCoord from HDF5"""
        ra = from_hdf5(hdf5_handle['ra'])
        dec = from_hdf5(hdf5_handle['dec'])

        if hdf5_handle.attrs['has_distance']:
            distance = from_hdf5(hdf5_handle['distance'])
            coord = SkyCoord(ra=ra, dec=dec, distance=distance, frame='icrs')
        else:
            coord = SkyCoord(ra=ra, dec=dec, frame='icrs')

        original_frame = hdf5_handle.attrs['frame']
        if original_frame != 'icrs':
            coord = coord.transform_to(original_frame)

        return coord
