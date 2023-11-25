#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DIR=$(realpath ${SCRIPT_DIR}/..)
cd ${DIR}/src/python
export CUDA_MODULE_LOADING=LAZY
export KMP_DUPLICATE_LIB_OK=TRUE

if [ "${CONDA_PREFIX}" != "" ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
fi

python -m enfugue.test.0_performance engine > engine.txt
python -m enfugue.test.0_performance manager > manager.txt
python -m enfugue.test.0_performance pipeline > pipeline.txt
