# An ASR microphone server using a faster whisper model sending output to MQTT

As simple ASR client for linux capturing audio from a local microphone using
gstreamer. The audio is analyzed using the [Silero Voice Activation
Detector](https://github.com/snakers4/silero-vad) (VAD). Identified segments
are transcribed with [Faster
Whisper](https://github.com/guillaumekln/faster-whisper), a reimplementation of
[OpenAI's Whisper](https://github.com/openai/whisper) model using
[CTranslate2](https://github.com/OpenNMT/CTranslate2/). The transcribed outputs
are send to a MQTT topic, so this client requires a running MQTT broker. For this, either install the mosquitto package (`apt install mosquitto`) or use a mosquitto docker image.

# Installation

## Ubuntu 22.04

These installation instructions are tested on Ubuntu 22.04, and do not require a virtual environment like venv or conda. Install python bindings for the gstreamer libraries, and the MQTT broker:

```
sudo apt install libgirepository1.0-dev python3-gst-1.0 libcairo2-dev mosquitto python3-pip git

pip install -r requirements.txt
```

After the installation, libcairo-dev and its dependencies can be removed:

```
sudo apt remove libcairo2-dev
sudo apt autoremove
```

- Download the VAD model and the Faster Whisper models (large-v2 by default)
  (faster whisper from [Huggingface](https://huggingface.co/guillaumekln))

```commandline
download-models.sh <optional-list-of-whisper-model-sizes>
```

## Ubuntu 24.04 and 24.10

Due to some changes in pip, the following setup is proposed. First install miniconda or conda according to the installation instructions given on their website.

Due to changes in package installation policies, installation is slightly different:

```
sudo apt install libgirepository1.0-dev python3-gst-1.0 libcairo2-dev mosquitto git

conda create -n whisper python=3.12 pip
conda activate whisper
pip install -r requirements.txt
```

## After package installation

Maybe you have to adapt the pipeline in `local_de_config.yaml`, or the language, the current default expects a ReSpeaker as default PulseAudio device. For the ReSpeaker, use the multichannel, not the analog-stereo.monoto device! You can check your local audio device configuration with

```
pacmd list-sources | grep -e 'index:' -e device.string -e 'name:'
```

Also, check the path to the cuda native libraries for `LD_LIBRARY_PATH` in the `run_whisper.sh` script in case of problems finding the `.so` libraries.

To set the default source to the ReSpeaker, use:

```
pacmd set-default-source 'alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.multichannel-input'
```

Check the content of gstmicpipeline.py in case of problems. In the audio directory, the microphone audio is stored in asrmon-XX.wav files and the data transferred to the ASR in chunk-XX.wav

```
./run_whisper.sh local_de_config.yaml
```

The ASR result will be send to the `whisperasr/asrresult/<lang>` MQTT topic.
