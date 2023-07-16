#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DIR=$(realpath ${SCRIPT_DIR}/..)
cd ${DIR}/src/python
export CUDA_MODULE_LOADING=LAZY
if [ $# -gt 0 ]; then
    CONFIG=$1
    shift
else
    CONFIG=server
fi
python -m pibble server ../../config/development/$CONFIG.yml $@
