#!/bin/bash
scrdir=`dirname $0`
cd "$scrdir"
uv run python -u $scrdir/src/vosk_transcriptor.py "$@"
