#!/usr/bin/env bash
ulimit -n unlimited 2>/dev/null >/dev/null || true
echo "Starting Enfugue server. Press Ctrl+C to exit."
python3 -m enfugue run $@
echo "Goodbye!"
