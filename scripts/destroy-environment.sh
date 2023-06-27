#!/usr/bin/env bash
SCRIPT_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
NAME=$(cat ${SCRIPT_DIR}/../environment.yaml | grep 'name: ' | awk -F': ' '{print $2}')
CONDA_HOME=$(realpath $(dirname $(which conda))/..)
ENV_CHECK=$(conda env list | grep ${NAME})

if [ "${ENV_CHECK}" == "" ]; then
    echo "${NAME} environment does not exist"
    exit 2
else
    echo "Removing ${NAME} environment"
    yes | conda remove -n ${NAME} --all
    exit 0
fi
