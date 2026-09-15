#!/bin/bash
scrdir=`dirname $0`
uv --project="$scrdir" run python -u $scrdir/src/vosk_transcriptor.py "$@"
