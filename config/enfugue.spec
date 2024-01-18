# -*- mode: python ; coding: utf-8 -*-
# vim: set syntax=python:
import os
import sys

sys.setrecursionlimit(sys.getrecursionlimit() * 10)

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    collect_all,
    copy_metadata,
)

here = os.path.abspath(os.getcwd())

#
# Start build configurables
#

source_script = os.path.join(here, 'src', 'python', 'enfugue', 'enfugue.py')
icon_path = os.path.join(here, 'src', 'img', 'favicon', 'favicon-64x64.png')

name = 'enfugue-server'
debug = False
console = False

all_packages = ['torch', 'jaxlib', 'onnxruntime', 'transformers', 'basicsr', 'gfpgan', 'skimage', 'realesrgan', 'tensorrt', 'tensorrt_libs', 'tensorrt-libs', 'PIL', 'pyarrow', 'huggingface_hub', 'huggingface-hub', 'pyyaml', 'accelerate', 'pydantic', 'safetensors', 'timm', 'diffusers', 'certifi', 'mmengine', 'mmcv', 'mmdet', 'mmpose', 'yapf', 'yapf_third_party', 'sentencepiece', 'enfugue', 'cv2', 'opencv-python', 'opencv-python-headless', 'open-clip-torch', 'open_clip', 'xformers', 'compel']
data_packages = []
submodule_packages = []
metadata_packages = ['requests', 'tqdm', 'numpy', 'tokenizers', 'importlib_metadata', 'regex', 'packaging', 'filelock', 'cheroot', 'pillow', 'enfugue', 'requests', 'scipy', 'beautifulsoup4']
hidden_packages = ['pkg_resources.py2_warn', 'pytorch', 'requests', 'jax', 'cheroot.ssl', 'cheroot.ssl.builtin', 'pillow']


#
# End build configurables
#

datas, binaries = [], []

for package in data_packages:
    datas += collect_data_files(package)

for package in metadata_packages:
    datas += copy_metadata(package)

for package in submodule_packages:
    hidden_packages += collect_submodules(package)

for package in all_packages:
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hidden_packages += pkg_hidden


block_cipher = None

a = Analysis(
    [source_script],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_packages,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=name,
    debug=debug,
    icon=icon_path,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=console,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=name,
)
