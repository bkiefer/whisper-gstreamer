#!/bin/sh
#set -x
mkdir audio > /dev/null 2>&1

if test -f silero_vad.jit; then
    :
else
    wget https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.jit
fi

download_models() {
    python -c "from faster_whisper import utils;import sys
for model in sys.argv[1:]:
    utils.download_model(model, output_dir='./whisper-models/' + model)" "$*"
}

if test -z "$1"; then
    download_models large-v2
else
    download_models "$*"
fi
