#!/usr/bin/env bash
SCRIPT_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
ROOT_DIR=$(realpath ${SCRIPT_DIR}/..)
if [ "${OSTYPE}" = "cygwin" ]; then
    ROOT_DIR=/cygwin64${ROOT_DIR}
fi
FAVICON_DIR=${ROOT_DIR}/src/img/favicon/

if ! command -v convert &> /dev/null; then
    echo "'convert' command not found. Install imagemagick to get it."
    exit 5
fi

if [ $# -lt 1 ]; then
    echo "USAGE: install-favicon.sh <base-image>"
    exit 3
fi

BASE_IMAGE=$1
IMAGE_SIZE=$(identify ${BASE_IMAGE} | awk '{print $3}')

if [ "${IMAGE_SIZE}" != "256x256" ]; then
    echo "Wrong size. Image should be 256x256, got ${IMAGE_SIZE} instead."
    exit 2
fi

SIZES=(256 128 64 32 16)

for SIZE in ${SIZES[@]}; do
    echo "Saving ${SIZE}x${SIZE}"
    convert ${BASE_IMAGE} -resize ${SIZE} ${FAVICON_DIR}/favicon-${SIZE}x${SIZE}.png
done

echo "Saving .ico"
convert ${BASE_IMAGE} -resize 16x16 ${FAVICON_DIR}/favicon.ico

echo "Installed favicon $(basename ${BASE_IMAGE})"
