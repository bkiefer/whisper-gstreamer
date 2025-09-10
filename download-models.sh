#!/bin/sh
#set -x
mkdir audio > /dev/null 2>&1

if test -f models/silero_vad.jit; then
    :
else
    cd models
    wget https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.jit
    cd ..
fi

download_models() {
    mkdir -p models/whisper 2>/dev/null
    python -c "from faster_whisper import utils;import sys
for model in sys.argv[1:]:
    utils.download_model(model, output_dir='./models/whisper' + model)" "$*"
}

if test -z "$1"; then
    download_models large-v3-turbo
else
    download_models "$*"
fi
