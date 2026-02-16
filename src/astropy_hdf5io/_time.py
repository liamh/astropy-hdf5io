"""Serialization support for astropy.time.Time"""

from fsc.hdf5_io import subscribe_hdf5
from astropy.time import Time
import numpy as np


def _time_to_hdf5(self, hdf5_handle):
    """Serialize Time to HDF5"""
    hdf5_handle['type_tag'] = 'astropy_hdf5io.Time'

    jd_values = self.jd
    if isinstance(jd_values, np.ndarray):
        hdf5_handle.create_dataset('jd', data=jd_values)
    else:
        hdf5_handle['jd'] = jd_values

    hdf5_handle['jd'].attrs['scale'] = self.scale
    hdf5_handle['jd'].attrs['format'] = self.format


Time.to_hdf5 = _time_to_hdf5


@subscribe_hdf5('astropy_hdf5io.Time', check_on_load=False)
class _TimeDeserializer:
    """Deserializer for Time"""

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        """Deserialize Time from HDF5"""
        jd = hdf5_handle['jd'][()]
        scale = hdf5_handle['jd'].attrs['scale']
        return Time(jd, format='jd', scale=scale)
