#!/usr/bin/env bash
PYTHON=python3.10
ulimit -n unlimited 2>/dev/null >/dev/null || true
echo "Starting Enfugue server. Press Ctrl+C to exit."
$PYTHON -m enfugue run $@
echo "Goodbye!"
