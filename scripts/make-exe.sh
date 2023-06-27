#!/usr/bin/env bash
if [ $# -lt 1 ]; then
    echo "USAGE: make-exe.sh <main_script.py>"
    exit 5
fi

MAIN_SCRIPT=$1
BASE_NAME=$(basename ${MAIN_SCRIPT})
PKG_NAME=${BASE_NAME%.*}

# Install dependencies for building
python -m pip install pyinstaller flax\>=0.5,\<0.6 jax==0.3.25 jaxlib==0.3.25 -f https://whls.blob.core.windows.net/unstable/index.html

if ! command -v pyinstaller &> /dev/null; then
    echo "pyinstaller not found after installation"
    exit 3
fi

export KMP_DUPLICATE_LIB_OK=TRUE 

pyinstaller ${MAIN_SCRIPT} --onedir --hidden-import pkg_resources.py2_warn --hidden-import pytorch --collect-data torch --copy-metadata torch --hidden-import requests --copy-metadata requests --copy-metadata tqdm --copy-metadata numpy --copy-metadata tokenizers --copy-metadata importlib_metadata --copy-metadata regex --copy-metadata packaging --copy-metadata filelock --hidden-import jax --hidden-import jaxlib --collect-data jaxlib --copy-metadata jaxlib --hidden-import transformers --collect-data transformers --copy-metadata transformers --hidden-import transformers.models --hidden-import transformers.models.blip

touch ./dist/${PKG_NAME}/transformers/__init__.py

echo "Wrote ./dist/${PKG_NAME}/${PKG_NAME}.exe"
