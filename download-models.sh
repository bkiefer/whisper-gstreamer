#!/bin/sh
mkdir audio > /dev/null 2>&1

if test -f silero_vad.jit; then
    :
else
    wget https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.jit
fi

if test -z "$1"; then
    $1="large-v2"
fi

python -c "from faster_whisper import utils
import sys
for model in sys.argv[1:]:
  utils.download_model(model, output_dir='./whisper-models/' + model)" "$*"
