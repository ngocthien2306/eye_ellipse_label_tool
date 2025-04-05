import os
import numpy
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

numpy_dir = r'C:\Users\ngoct\envs\event\Lib\site-packages\numpy'
numpy_core_dir = os.path.join(numpy_dir, 'core')

hiddenimports = [
    'numpy',
    'numpy.core._multiarray_umath',
    'numpy.core.multiarray',
    'numpy.core.numeric',
    'numpy.core._dtype_ctypes',
    'numpy.lib.format',
    'numpy.core._multiarray_tests',
    'numpy.core.numerictypes',
    'numpy.core.fromnumeric',
    'numpy.core.arrayprint'
]

binaries = []
core_libs = []

# Collect all PYD files from numpy.core
for f in os.listdir(numpy_core_dir):
    if f.endswith('.pyd'):
        full_path = os.path.join(numpy_core_dir, f)
        rel_path = os.path.join('numpy', 'core')
        core_libs.append((full_path, rel_path))

binaries.extend(core_libs)
binaries.extend(collect_dynamic_libs('numpy'))
datas = collect_data_files('numpy')