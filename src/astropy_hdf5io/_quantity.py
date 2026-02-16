"""Serialization support for astropy.units.Quantity"""

import numpy as np
from fsc.hdf5_io import subscribe_hdf5
import astropy.units as u


# Monkey patch to_hdf5 method directly onto Quantity class
def _quantity_to_hdf5(self, hdf5_handle):
    """Serialize Quantity to HDF5"""
    # Store type_tag as a DATASET, not an attribute (fsc.hdf5-io requirement)
    hdf5_handle['type_tag'] = 'astropy_hdf5io.Quantity'

    if isinstance(self.value, np.ndarray):
        hdf5_handle.create_dataset('value', data=self.value)
    else:
        hdf5_handle['value'] = self.value

    # Store unit as attribute on the value dataset
    hdf5_handle['value'].attrs['unit'] = str(self.unit)

    # Check if it's a special unit type (like logarithmic units)
    unit_type = type(self.unit).__name__
    hdf5_handle['value'].attrs['unit_type'] = unit_type


u.Quantity.to_hdf5 = _quantity_to_hdf5


@subscribe_hdf5('astropy_hdf5io.Quantity', check_on_load=False)
class _QuantityDeserializer:
    """Deserializer for Quantity"""

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        """Deserialize Quantity from HDF5"""
        value = hdf5_handle['value'][()]
        unit_str = hdf5_handle['value'].attrs['unit']

        # Handle different unit types
        unit_type = hdf5_handle['value'].attrs.get('unit_type', 'Unit')

        if unit_type == 'MagUnit' or unit_type == 'DecibelUnit':
            # For logarithmic units, we need to use the physical unit
            unit = u.Unit(unit_str)
            # Create a Quantity with the physical unit, which will get the right type
            from astropy.units import Magnitude, Decibel
            if 'mag' in unit_str.lower():
                return Magnitude(value, unit)
            elif 'dB' in unit_str or 'dex' in unit_str:
                return Decibel(value, unit)
            else:
                # Fallback
                return value * unit
        else:
            unit = u.Unit(unit_str)
            return u.Quantity(value, unit=unit)
