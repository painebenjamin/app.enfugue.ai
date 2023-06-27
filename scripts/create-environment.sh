#!/usr/bin/env bash
SCRIPT_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
DIR=$(realpath ${SCRIPT_DIR}/..)
NAME=$(cat ${DIR}/environment.yaml | grep 'name: ' | awk -F': ' '{print $2}')
CONDA_HOME=$(realpath $(dirname $(which conda))/..)
ENV_CHECK=$(conda env list | grep ${NAME})

if [ "${ENV_CHECK}" = "" ]; then
    echo "Creating ${NAME} environment"
    conda env create -f ${DIR}/environment.yaml
fi

source ${CONDA_HOME}/bin/activate ${NAME}

mkdir -p ${SCRIPT_DIR}/.install
cd ${SCRIPT_DIR}/.install

pip install -r ${DIR}/src/python/enfugue/requirements.txt -I

if [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    pip install -r ${DIR}/src/python/enfugue/tensorrt-requirements.txt -I
fi

cd
rm -rf ${SCRIPT_DIR}/.install
