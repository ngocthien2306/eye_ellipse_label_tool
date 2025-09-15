# Building EVS Label Executable

This document provides instructions for building a standalone executable version of the EVS Label application using PyInstaller.

## Prerequisites

- Python 3.8+ with all dependencies installed
- PyInstaller: `pip install pyinstaller`
- All required dependencies from `requirements.txt`

## PyInstaller Build Command

### Windows

```batch
pyinstaller --noconfirm --onefile --clean ^
--name "EVS Label" ^
--additional-hooks-dir="hooks" ^
--hidden-import cv2 ^
--hidden-import albumentations ^
--hidden-import numpy.core._multiarray_umath ^
--hidden-import numpy.core.multiarray ^
--hidden-import numpy.core._dtype_ctypes ^
--hidden-import psutil ^
--hidden-import metavision_hal ^
--hidden-import metavision_core ^
--hidden-import metavision_core.event_io ^
--hidden-import metavision_core.event_io.raw_reader ^
--collect-all metavision_hal ^
--collect-all metavision_core ^
--exclude-module cutensor ^
--exclude-module torch ^
--exclude-module tensorflow ^
--exclude-module tensorflow-gpu ^
--exclude-module tensorboard ^
--exclude-module keras ^
--exclude-module cudnn ^
--exclude-module notebook ^
--exclude-module ipykernel ^
--exclude-module jedi ^
main.py
```

### Linux/Mac

```bash
pyinstaller --noconfirm --onefile --clean \
--name "EVS Label" \
--additional-hooks-dir="hooks" \
--hidden-import cv2 \
--hidden-import albumentations \
--hidden-import numpy.core._multiarray_umath \
--hidden-import numpy.core.multiarray \
--hidden-import numpy.core._dtype_ctypes \
--hidden-import psutil \
--hidden-import metavision_hal \
--hidden-import metavision_core \
--hidden-import metavision_core.event_io \
--hidden-import metavision_core.event_io.raw_reader \
--collect-all metavision_hal \
--collect-all metavision_core \
--exclude-module cutensor \
--exclude-module torch \
--exclude-module tensorflow \
--exclude-module tensorflow-gpu \
--exclude-module tensorboard \
--exclude-module keras \
--exclude-module cudnn \
--exclude-module notebook \
--exclude-module ipykernel \
--exclude-module jedi \
main.py
```

## Required Hook Files

### Setting Up Hook Files

1. Create a `hooks` directory in your project folder
2. Create the following hook files in that directory:

### hook-cv2.py

```python
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

hiddenimports = ['numpy']
binaries = collect_dynamic_libs('cv2')
datas = collect_data_files('cv2')
```

### hook-numpy.py

```python
import os
import numpy
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

# Adjust this path to match your Python environment
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
```

## PyInstaller Options Explained

| Option | Description |
|--------|-------------|
| `--noconfirm` | Replaces output directory without asking for confirmation |
| `--onefile` | Creates a single executable file instead of a directory |
| `--clean` | Cleans PyInstaller cache and removes temporary files |
| `--name "EVS Label"` | Sets the name of the output executable |
| `--additional-hooks-dir="hooks"` | Directory containing additional hooks |
| `--hidden-import` | Includes modules that PyInstaller can't automatically detect |
| `--collect-all` | Collects all submodules, data files, and binaries for the specified package |
| `--exclude-module` | Excludes unnecessary modules to reduce executable size |

## Common Issues and Solutions

### Missing Modules

If you encounter "ModuleNotFoundError" when running the executable, add the missing module to the `--hidden-import` list:

```
--hidden-import missing_module_name
```

### DLL Loading Failed

If you see "DLL loading failed" errors, ensure that you've included all required DLLs:

```
--add-binary "path/to/required.dll;."
```

### Large Executable Size

The current configuration excludes several large packages (PyTorch, TensorFlow, etc.) to reduce the size. If you need to reduce it further, consider:

1. Adding more packages to the `--exclude-module` list
2. Using `--onedir` instead of `--onefile` (creates a directory instead of a single file)
3. Using UPX compression: `--upx-dir=/path/to/upx`

## After Building

After running the PyInstaller command:

1. The executable will be created in the `dist` directory
2. Test the executable to ensure it works properly
3. If it fails, check the console output for error messages

## Distribution

When distributing the executable:

1. Include any necessary data files or resources
2. Verify it works on a clean system without Python installed
3. Consider providing the version number in the filename