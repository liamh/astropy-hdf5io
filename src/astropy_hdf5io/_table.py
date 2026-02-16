"""Serialization support for astropy.table types"""

import numpy as np
from fsc.hdf5_io import subscribe_hdf5, to_hdf5, from_hdf5
from astropy.table import Table, QTable, Column, MaskedColumn
from astropy.timeseries import TimeSeries
from astropy.time import Time
import astropy.units as u


# ============================================================================
# Table
# ============================================================================

def _table_to_hdf5(self, hdf5_handle):
    """Serialize Table to HDF5"""
    hdf5_handle['type_tag'] = 'astropy_hdf5io.Table'

    # Store column names as attribute
    hdf5_handle.attrs['colnames'] = [str(name) for name in self.colnames]

    # Store each column
    columns_group = hdf5_handle.create_group('columns')
    for colname in self.colnames:
        col = self[colname]
        col_group = columns_group.create_group(colname)

        # Store column data - handle string columns specially
        if isinstance(col, MaskedColumn):
            col_group.attrs['masked'] = True
            # Convert unicode strings to bytes for HDF5 compatibility
            if col.dtype.kind == 'U':
                col_group.create_dataset('data', data=np.array(col.data, dtype='S'))
                col_group.attrs['was_unicode'] = True
            else:
                col_group.create_dataset('data', data=col.data)
                col_group.attrs['was_unicode'] = False
            col_group.create_dataset('mask', data=col.mask)
        else:
            col_group.attrs['masked'] = False
            # Convert unicode strings to bytes for HDF5 compatibility
            if col.dtype.kind == 'U':
                col_group.create_dataset('data', data=np.array(col.data, dtype='S'))
                col_group.attrs['was_unicode'] = True
            else:
                col_group.create_dataset('data', data=col.data)
                col_group.attrs['was_unicode'] = False

        # Store column metadata
        col_group.attrs['dtype'] = str(col.dtype)
        if col.unit is not None:
            col_group.attrs['unit'] = str(col.unit)
        else:
            col_group.attrs['unit'] = ''

        if col.description is not None:
            col_group.attrs['description'] = str(col.description)

        if col.format is not None:
            col_group.attrs['format'] = str(col.format)

        # Store column metadata dict
        if col.meta:
            meta_group = col_group.create_group('meta')
            for key, value in col.meta.items():
                try:
                    to_hdf5(value, meta_group.create_group(str(key)))
                except (TypeError, ValueError):
                    # Fallback for non-serializable metadata
                    meta_group.attrs[str(key)] = str(value)

    # Store table metadata
    if self.meta:
        meta_group = hdf5_handle.create_group('meta')
        for key, value in self.meta.items():
            try:
                to_hdf5(value, meta_group.create_group(str(key)))
            except (TypeError, ValueError):
                # Fallback for non-serializable metadata
                meta_group.attrs[str(key)] = str(value)


Table.to_hdf5 = _table_to_hdf5


@subscribe_hdf5('astropy_hdf5io.Table', check_on_load=False)
class _TableDeserializer:
    """Deserializer for Table"""

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        """Deserialize Table from HDF5"""
        colnames = list(hdf5_handle.attrs['colnames'])
        columns_group = hdf5_handle['columns']

        # Reconstruct columns
        columns = []
        for colname in colnames:
            col_group = columns_group[colname]
            data = col_group['data'][()]

            # Convert bytes back to unicode if needed
            if col_group.attrs.get('was_unicode', False):
                data = np.array(data, dtype='U')

            # Handle masked columns
            if col_group.attrs['masked']:
                mask = col_group['mask'][()]
                col = MaskedColumn(data=data, mask=mask, name=colname)
            else:
                col = Column(data=data, name=colname)

            # Restore column metadata
            if col_group.attrs['unit']:
                col.unit = u.Unit(col_group.attrs['unit'])

            if 'description' in col_group.attrs:
                col.description = col_group.attrs['description']

            if 'format' in col_group.attrs:
                try:
                    col.format = col_group.attrs['format']
                except (AttributeError, ValueError):
                    pass

            # Restore column meta dict
            if 'meta' in col_group:
                col.meta = {}
                meta_group = col_group['meta']
                for key in meta_group.keys():
                    try:
                        col.meta[key] = from_hdf5(meta_group[key])
                    except:
                        pass
                # Also get attrs
                for key in meta_group.attrs.keys():
                    if key not in col.meta:
                        col.meta[key] = meta_group.attrs[key]

            columns.append(col)

        # Create table
        table = Table(columns)

        # Restore table metadata
        if 'meta' in hdf5_handle:
            table.meta = {}
            meta_group = hdf5_handle['meta']
            for key in meta_group.keys():
                try:
                    table.meta[key] = from_hdf5(meta_group[key])
                except:
                    pass
            # Also get attrs
            for key in meta_group.attrs.keys():
                if key not in table.meta:
                    table.meta[key] = meta_group.attrs[key]

        return table


# ============================================================================
# QTable
# ============================================================================

def _qtable_to_hdf5(self, hdf5_handle):
    """Serialize QTable to HDF5"""
    hdf5_handle['type_tag'] = 'astropy_hdf5io.QTable'

    # Store column names
    hdf5_handle.attrs['colnames'] = [str(name) for name in self.colnames]

    # Store each column
    columns_group = hdf5_handle.create_group('columns')
    for colname in self.colnames:
        col = self[colname]
        col_group = columns_group.create_group(colname)

        # Check if column is a Time object
        if isinstance(col, Time):
            col_group.attrs['is_time'] = True
            to_hdf5(col, col_group.create_group('time'))
        # Check if column is a Quantity
        elif isinstance(col, u.Quantity):
            col_group.attrs['is_time'] = False
            col_group.attrs['is_quantity'] = True
            # Store the description separately since Quantity doesn't preserve it
            if hasattr(col, 'info') and hasattr(col.info, 'description') and col.info.description:
                col_group.attrs['description'] = str(col.info.description)
            to_hdf5(col, col_group.create_group('quantity'))

            # Store info.meta if it exists
            if hasattr(col, 'info') and hasattr(col.info, 'meta') and col.info.meta:
                info_meta_group = col_group.create_group('info_meta')
                for key, value in col.info.meta.items():
                    try:
                        to_hdf5(value, info_meta_group.create_group(str(key)))
                    except (TypeError, ValueError):
                        info_meta_group.attrs[str(key)] = str(value)
        else:
            col_group.attrs['is_time'] = False
            col_group.attrs['is_quantity'] = False
            # Regular column
            if isinstance(col, MaskedColumn):
                col_group.attrs['masked'] = True
                # Handle unicode strings
                if col.dtype.kind == 'U':
                    col_group.create_dataset('data', data=np.array(col.data, dtype='S'))
                    col_group.attrs['was_unicode'] = True
                else:
                    col_group.create_dataset('data', data=col.data)
                    col_group.attrs['was_unicode'] = False
                col_group.create_dataset('mask', data=col.mask)
            else:
                col_group.attrs['masked'] = False
                # Handle unicode strings
                if col.dtype.kind == 'U':
                    col_group.create_dataset('data', data=np.array(col.data, dtype='S'))
                    col_group.attrs['was_unicode'] = True
                else:
                    col_group.create_dataset('data', data=col.data)
                    col_group.attrs['was_unicode'] = False

            col_group.attrs['dtype'] = str(col.dtype)

        # Store column metadata (common for all types)
        if hasattr(col, 'description') and col.description is not None:
            col_group.attrs['description'] = str(col.description)

        if hasattr(col, 'format') and col.format is not None:
            col_group.attrs['format'] = str(col.format)

        # Store column metadata dict
        if hasattr(col, 'meta') and col.meta:
            meta_group = col_group.create_group('meta')
            for key, value in col.meta.items():
                try:
                    to_hdf5(value, meta_group.create_group(str(key)))
                except (TypeError, ValueError):
                    meta_group.attrs[str(key)] = str(value)

    # Store table metadata
    if self.meta:
        meta_group = hdf5_handle.create_group('meta')
        for key, value in self.meta.items():
            try:
                to_hdf5(value, meta_group.create_group(str(key)))
            except (TypeError, ValueError):
                meta_group.attrs[str(key)] = str(value)


QTable.to_hdf5 = _qtable_to_hdf5


@subscribe_hdf5('astropy_hdf5io.QTable', check_on_load=False)
class _QTableDeserializer:
    """Deserializer for QTable"""

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        """Deserialize QTable from HDF5"""
        colnames = list(hdf5_handle.attrs['colnames'])
        columns_group = hdf5_handle['columns']

        # Reconstruct columns
        columns = []
        for colname in colnames:
            col_group = columns_group[colname]

            # Handle Time columns
            if col_group.attrs.get('is_time', False):
                col = from_hdf5(col_group['time'])
                col = Column(col, name=colname)
            # Handle Quantity columns
            elif col_group.attrs.get('is_quantity', False):
                col = from_hdf5(col_group['quantity'])
                # Wrap in Column to preserve name and metadata
                col = Column(col, name=colname)
                # Restore description if it was saved
                if 'description' in col_group.attrs:
                    col.info.description = col_group.attrs['description']

                # Restore info.meta if it was saved
                if 'info_meta' in col_group:
                    info_meta = {}
                    info_meta_group = col_group['info_meta']
                    for key in info_meta_group.keys():
                        try:
                            info_meta[key] = from_hdf5(info_meta_group[key])
                        except:
                            pass
                    for key in info_meta_group.attrs.keys():
                        if key not in info_meta:
                            info_meta[key] = info_meta_group.attrs[key]
                    col.info.meta = info_meta
            else:
                # Regular column
                data = col_group['data'][()]

                # Convert bytes back to unicode if needed
                if col_group.attrs.get('was_unicode', False):
                    data = np.array(data, dtype='U')

                if col_group.attrs.get('masked', False):
                    mask = col_group['mask'][()]
                    col = MaskedColumn(data=data, mask=mask, name=colname)
                else:
                    col = Column(data=data, name=colname)

            # Restore column metadata
            if 'description' in col_group.attrs:
                try:
                    col.description = col_group.attrs['description']
                except AttributeError:
                    # Some column types (like Quantity) don't support description
                    pass

            if 'format' in col_group.attrs:
                try:
                    col.format = col_group.attrs['format']
                except (AttributeError, ValueError):
                    # Some column types don't support format or the format is incompatible
                    pass

            # Restore column meta dict
            if 'meta' in col_group:
                col.meta = {}
                meta_group = col_group['meta']
                for key in meta_group.keys():
                    try:
                        col.meta[key] = from_hdf5(meta_group[key])
                    except:
                        pass
                for key in meta_group.attrs.keys():
                    if key not in col.meta:
                        col.meta[key] = meta_group.attrs[key]

            columns.append(col)

        # Create QTable
        qtable = QTable(columns)

        # Restore table metadata
        if 'meta' in hdf5_handle:
            qtable.meta = {}
            meta_group = hdf5_handle['meta']
            for key in meta_group.keys():
                try:
                    qtable.meta[key] = from_hdf5(meta_group[key])
                except:
                    pass
            for key in meta_group.attrs.keys():
                if key not in qtable.meta:
                    qtable.meta[key] = meta_group.attrs[key]

        return qtable


# ============================================================================
# TimeSeries
# ============================================================================

def _timeseries_to_hdf5(self, hdf5_handle):
    """Serialize TimeSeries to HDF5"""
    hdf5_handle['type_tag'] = 'astropy_hdf5io.TimeSeries'

    # Find the time column name - it's the column containing the Time object
    time_colname = None
    for colname in self.colnames:
        if isinstance(self[colname], Time):
            time_colname = colname
            break

    if time_colname is None:
        # Fallback - try common time column names
        for possible_name in ['time', 'Time', 'TIME', 'date', 'Date']:
            if possible_name in self.colnames:
                time_colname = possible_name
                break

    if time_colname is None:
        raise ValueError("Could not find time column in TimeSeries")

    hdf5_handle.attrs['time_column'] = time_colname

    # Store time column separately
    time_group = hdf5_handle.create_group('time')
    to_hdf5(self[time_colname], time_group.create_group('time_data'))

    # Store other column names
    other_colnames = [name for name in self.colnames if name != time_colname]
    hdf5_handle.attrs['colnames'] = other_colnames

    # Store each data column
    if other_colnames:
        columns_group = hdf5_handle.create_group('columns')
        for colname in other_colnames:
            col = self[colname]
            col_group = columns_group.create_group(colname)

            # Check if column is a Quantity
            if isinstance(col, u.Quantity):
                col_group.attrs['is_quantity'] = True
                to_hdf5(col, col_group.create_group('quantity'))
            else:
                col_group.attrs['is_quantity'] = False
                if isinstance(col, MaskedColumn):
                    col_group.attrs['masked'] = True
                    # Handle unicode strings
                    if col.dtype.kind == 'U':
                        col_group.create_dataset('data', data=np.array(col.data, dtype='S'))
                        col_group.attrs['was_unicode'] = True
                    else:
                        col_group.create_dataset('data', data=col.data)
                        col_group.attrs['was_unicode'] = False
                    col_group.create_dataset('mask', data=col.mask)
                else:
                    col_group.attrs['masked'] = False
                    # Handle unicode strings
                    if col.dtype.kind == 'U':
                        col_group.create_dataset('data', data=np.array(col.data, dtype='S'))
                        col_group.attrs['was_unicode'] = True
                    else:
                        col_group.create_dataset('data', data=col.data)
                        col_group.attrs['was_unicode'] = False

                col_group.attrs['dtype'] = str(col.dtype)

            # Store column metadata
            if hasattr(col, 'unit') and col.unit is not None:
                col_group.attrs['unit'] = str(col.unit)

            if hasattr(col, 'description') and col.description is not None:
                col_group.attrs['description'] = str(col.description)

            if hasattr(col, 'format') and col.format is not None:
                col_group.attrs['format'] = str(col.format)

            # Store column metadata dict
            if hasattr(col, 'meta') and col.meta:
                meta_group = col_group.create_group('meta')
                for key, value in col.meta.items():
                    try:
                        to_hdf5(value, meta_group.create_group(str(key)))
                    except (TypeError, ValueError):
                        meta_group.attrs[str(key)] = str(value)

    # Store table metadata
    if self.meta:
        meta_group = hdf5_handle.create_group('meta')
        for key, value in self.meta.items():
            try:
                to_hdf5(value, meta_group.create_group(str(key)))
            except (TypeError, ValueError):
                meta_group.attrs[str(key)] = str(value)


TimeSeries.to_hdf5 = _timeseries_to_hdf5


@subscribe_hdf5('astropy_hdf5io.TimeSeries', check_on_load=False)
class _TimeSeriesDeserializer:
    """Deserializer for TimeSeries"""

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        """Deserialize TimeSeries from HDF5"""
        # Restore time column
        time_data = from_hdf5(hdf5_handle['time']['time_data'])
        time_column_name = hdf5_handle.attrs['time_column']

        # Restore other columns
        other_colnames = list(hdf5_handle.attrs['colnames'])

        data = {}
        if other_colnames and 'columns' in hdf5_handle:
            columns_group = hdf5_handle['columns']

            for colname in other_colnames:
                col_group = columns_group[colname]

                # Handle Quantity columns
                if col_group.attrs['is_quantity']:
                    col = from_hdf5(col_group['quantity'])
                else:
                    # Regular column
                    col_data = col_group['data'][()]

                    # Convert bytes back to unicode if needed
                    if col_group.attrs.get('was_unicode', False):
                        col_data = np.array(col_data, dtype='U')

                    if col_group.attrs.get('masked', False):
                        mask = col_group['mask'][()]
                        col = MaskedColumn(data=col_data, mask=mask)
                    else:
                        col = Column(data=col_data)

                    # Restore unit if present
                    if 'unit' in col_group.attrs and col_group.attrs['unit']:
                        col = col * u.Unit(col_group.attrs['unit'])

                data[colname] = col

                # Restore column description if present
                if 'description' in col_group.attrs:
                    if isinstance(data[colname], Column):
                        data[colname].description = col_group.attrs['description']

        # Create TimeSeries
        ts = TimeSeries(time=time_data, data=data)

        # Restore table metadata
        if 'meta' in hdf5_handle:
            ts.meta = {}
            meta_group = hdf5_handle['meta']
            for key in meta_group.keys():
                try:
                    ts.meta[key] = from_hdf5(meta_group[key])
                except:
                    pass
            for key in meta_group.attrs.keys():
                if key not in ts.meta:
                    ts.meta[key] = meta_group.attrs[key]

        return ts
