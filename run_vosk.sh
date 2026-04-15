#!/bin/bash
scrdir=`dirname $0`
cd "$scrdir"
#python mqtt_micro_vadasr.py de_config.yml
if test -z "$1"; then
    conf="-c config_de.yml"
fi
uv run python -u $scrdir/src/vosk_transcriptor.py $conf "$@"
