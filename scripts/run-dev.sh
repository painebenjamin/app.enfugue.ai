#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DIR=$(realpath ${SCRIPT_DIR}/..)
cd ${DIR}/src/python
export CUDA_MODULE_LOADING=LAZY
python -m pibble server ../../config/development/server.yml $@
