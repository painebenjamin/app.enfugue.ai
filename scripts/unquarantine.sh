#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
find $SCRIPT_DIR -type f -exec xattr -dr com.apple.quarantine {} \;
