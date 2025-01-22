#!/bin/bash
#set -xe
if test -z "$1"; then
    echo "Usage: $0 <config.yml>"
    exit 1
fi
scrdir=`dirname "$0"`
docker run -it \
       -e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
       --add-host host.docker.internal=host-gateway \
       -v ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native \
       -v $HOME/.config/pulse/cookie:/root/.config/pulse/cookie \
       -v "$scrdir/$1":/app/config.yml \
       -v "$scrdir/whisper-models":/app/whisper-models \
       -v "$scrdir/audio":/app/audio \
       --gpus=all \
       --entrypoint /bin/bash \
       whisper_asr -c "./run_whisper.sh -c config.yml"
