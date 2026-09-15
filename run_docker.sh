#!/bin/bash
#set -xe
. utils.sh
if test -z "$1"; then
    echo "Usage: $0 <config.yml>"
    exit 1
fi
scrdir=`dirname "$0"`
if test -z "$run_script"; then
    run_script="./run_whisper.sh"
fi
config="$1"
shift
docker run --rm --name whisperasr \
       --device /dev/snd --group-add audio \
       -e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
       --add-host host.docker.internal:host-gateway \
       -v ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native \
       -v "$scrdir/${config}":/app/config.yml \
       -v "$scrdir/models":/app/models \
       -v "$scrdir/audio":/app/audio \
       -v "$scrdir/outputs":/app/outputs \
       -v "$scrdir/inputs":/app/inputs \
       --gpus=all \
       --entrypoint=/bin/bash \
       $(getimage) -c "$run_script -m -c config.yml"
