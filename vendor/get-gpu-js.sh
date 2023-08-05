#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
if [ $# -eq 1 ]; then
    ROOT=$1
else 
    ROOT=$DIR/../src
fi

JS=$ROOT/js
JS_TARGET=$JS/vendor/gpu
JS_FILE=gpu-browser.min.js
DIST=https://raw.githubusercontent.com/gpujs/gpu.js/develop/dist/$JS_FILE

cd $DIR

curl -sOL $DIST
rm -rf $JS_TARGET
mkdir -p $JS_TARGET
mv $JS_FILE $JS_TARGET

echo "Successfully retrieved GPU.js"
