#!/bin/bash
#set -xe
. utils.sh
if test -z "$1"; then
    echo "Usage: $0 <config.yml>"
    exit 1
fi
scrdir=`dirname "$0"`
config="$1"
shift
docker run --rm \
       --device /dev/snd --group-add audio \
       -e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
       --add-host host.docker.internal:host-gateway \
       -v ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native \
       -v $HOME/.config/pulse/cookie:/root/.config/pulse/cookie \
       -v "$scrdir/${config}":/app/config.yml \
       -v "$scrdir/models":/app/models \
       -v "$scrdir/audio":/app/audio \
       -v "$scrdir/outputs":/app/outputs \
       --gpus=all \
       --entrypoint=/bin/bash \
       $(getimage) -c "./run_whisper.sh -m -c config.yml"
