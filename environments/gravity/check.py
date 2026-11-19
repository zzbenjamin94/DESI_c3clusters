"""Small import and HDF5/FITS round-trip check; no catalogs required."""
import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import h5py
from astropy.table import Table


def main():
    for name in (
        'numpy', 'scipy', 'pandas', 'astropy', 'matplotlib', 'h5py',
        'healpy', 'emcee', 'corner', 'pygtc', 'schwimmbad', 'multiprocess',
        'ipykernel', 'jupyterlab',
    ):
        mod = importlib.import_module(name)
        print(name, getattr(mod, '__version__', 'import OK'))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        values = np.arange(4.0)
        Table({'test': values}).write(root / 'test.fits')
        np.testing.assert_array_equal(Table.read(root / 'test.fits')['test'], values)
        with h5py.File(root / 'test.h5', 'w') as handle:
            handle.create_dataset('test', data=values)
        with h5py.File(root / 'test.h5', 'r') as handle:
            np.testing.assert_array_equal(handle['test'][:], values)
    print('PASS: imports and FITS/HDF5 round trips.', sys.executable)


if __name__ == '__main__':
    main()
