#!/bin/sh
#set -x
here=`pwd`
scrdir=`dirname "$0"`
cd $scrdir
# Not using $scrdir since we assume the cd there!!

cd models
if test \! -f silero_vad.jit; then
    wget https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.jit
fi

model_root="kaldi_models"
vosk_model="vosk-model-de-tuda-0.6-900k"

# Download the german kaldi model
if test \! -d "$model_root"/"$vosk_model"; then
    mkdir "$model_root" 2>/dev/null
    cd "$model_root"
    wget https://alphacephei.com/vosk/models/"$vosk_model".zip
    unzip "$vosk_model".zip
    rm "$vosk_model".zip
fi
