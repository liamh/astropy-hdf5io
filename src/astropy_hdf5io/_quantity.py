"""Serialization support for astropy.units.Quantity"""

import json

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

    # Check if it's a special unit type (like logarithmic units)
    unit_type = type(self.unit).__name__
    hdf5_handle['value'].attrs['unit_type'] = unit_type

    # Store unit: handle StructuredUnit separately since str() is not parseable
    if isinstance(self.unit, u.StructuredUnit):
        field_units = {name: str(self.unit[name]) for name in self.unit.field_names}
        hdf5_handle['value'].attrs['unit'] = json.dumps(field_units)
    else:
        hdf5_handle['value'].attrs['unit'] = str(self.unit)


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

        if unit_type == 'StructuredUnit':
            field_units = json.loads(unit_str)
            unit = u.StructuredUnit(
                tuple(u.Unit(v) for v in field_units.values()),
                names=tuple(field_units.keys()),
            )
            return u.Quantity(value, unit=unit)
        elif unit_type == 'MagUnit' or unit_type == 'DecibelUnit':
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
