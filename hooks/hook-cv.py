from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

hiddenimports = ['numpy']
binaries = collect_dynamic_libs('cv2')
datas = collect_data_files('cv2')