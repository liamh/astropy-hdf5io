"""Serialization support for astropy.time.Time"""

from fsc.hdf5_io import subscribe_hdf5
from astropy.time import Time
import numpy as np


def _time_to_hdf5(self, hdf5_handle):
    """Serialize Time to HDF5"""
    hdf5_handle['type_tag'] = 'astropy_hdf5io.Time'

    # Store the internal two-part JD representation for maximum precision
    # Time internally stores as jd1 (integer part) and jd2 (fractional part)
    jd1_values = self.jd1
    jd2_values = self.jd2

    if isinstance(jd1_values, np.ndarray):
        hdf5_handle.create_dataset('jd1', data=jd1_values)
        hdf5_handle.create_dataset('jd2', data=jd2_values)
    else:
        hdf5_handle['jd1'] = jd1_values
        hdf5_handle['jd2'] = jd2_values

    # Store metadata
    hdf5_handle.attrs['scale'] = self.scale
    hdf5_handle.attrs['format'] = self.format

    # Store precision if it's explicitly set
    try:
        if hasattr(self, '_time') and hasattr(self._time, 'precision'):
            precision = self._time.precision
            if precision is not None and 0 <= precision <= 9:
                hdf5_handle.attrs['precision'] = int(precision)
    except (AttributeError, ValueError):
        pass

    # Store location if available
    if self.location is not None:
        hdf5_handle.attrs['has_location'] = True
        from fsc.hdf5_io import to_hdf5
        to_hdf5(self.location, hdf5_handle.create_group('location'))
    else:
        hdf5_handle.attrs['has_location'] = False


Time.to_hdf5 = _time_to_hdf5


@subscribe_hdf5('astropy_hdf5io.Time', check_on_load=False)
class _TimeDeserializer:
    """Deserializer for Time"""

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        """Deserialize Time from HDF5"""
        # Check for new format (jd1/jd2) or old format (mjd)
        if 'jd1' in hdf5_handle and 'jd2' in hdf5_handle:
            jd1 = hdf5_handle['jd1'][()]
            jd2 = hdf5_handle['jd2'][()]
            scale = hdf5_handle.attrs['scale']
            original_format = hdf5_handle.attrs['format']

            # Load location if present
            location = None
            if hdf5_handle.attrs.get('has_location', False):
                from fsc.hdf5_io import from_hdf5
                location = from_hdf5(hdf5_handle['location'])

            # Get precision if it was saved
            precision = None
            if 'precision' in hdf5_handle.attrs:
                precision = int(hdf5_handle.attrs['precision'])

            # Create Time object using the two-part JD
            # This preserves maximum precision
            t = Time(jd1, jd2, format='jd', scale=scale, location=location)

            # Set the format to the original format
            if original_format != 'jd':
                t.format = original_format

            # Set precision if it was saved
            if precision is not None:
                t.precision = precision

            return t

        # Fallback for old format (mjd) - for backward compatibility
        elif 'mjd' in hdf5_handle:
            mjd = hdf5_handle['mjd'][()]
            scale = hdf5_handle['mjd'].attrs['scale']
            original_format = hdf5_handle['mjd'].attrs['format']

            location = None
            if hdf5_handle['mjd'].attrs.get('has_location', False):
                from fsc.hdf5_io import from_hdf5
                location = from_hdf5(hdf5_handle['location'])

            precision = None
            if 'precision' in hdf5_handle['mjd'].attrs:
                precision = int(hdf5_handle['mjd'].attrs['precision'])

            t = Time(mjd, format='mjd', scale=scale, location=location)

            if original_format != 'mjd':
                t.format = original_format

            if precision is not None:
                t.precision = precision

            return t

        else:
            raise ValueError("Unknown Time storage format in HDF5 file")
