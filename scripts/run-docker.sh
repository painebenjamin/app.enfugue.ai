#!/usr/bin/env sh
LOCAL_CACHE_DIR=$(realpath ~/.cache)
CONTAINER_CACHE_DIR=/opt/enfugue
PORT=45554
IMAGE=ghcr.io/painebenjamin/app.enfugue.ai:latest
docker run --rm -it --gpus all --runtime=nvidia -v ${LOCAL_CACHE_DIR}:${CONTAINER_CACHE_DIR} -p ${PORT}:${PORT} -e LOGGING_LEVEL='DEBUG' ${IMAGE} run
