#!/usr/bin/env sh
CACHE_DIR=$(realpath ~/.cache)
PORT=45554
IMAGE=enfugue-tensorrt
docker run --rm --gpus all --runtime=nvidia -v ${CACHE_DIR}:/home/enfugue/.cache -p ${PORT}:${PORT} -e LOGGING_LEVEL='DEBUG' ${IMAGE} run
