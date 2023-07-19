#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

export PATH=${SCRIPT_DIR}/torch/lib:${SCRIPT_DIR}/tensorrt:${PATH}
export LD_LIBRARY_PATH=${SCRIPT_DIR}/torch/lib:${SCRIPT_DIR}/tensorrt:${LD_LIBRARY_PATH}
export CUDA_MODULE_LOADING=LAZY
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1

echo "Starting Enfugue server. Press Ctrl+C to exit."
${SCRIPT_DIR}/enfugue-server $@
echo "Goodbye!"
