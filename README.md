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

## Ubuntu 22.04 or higher

These installation instructions are tested on Ubuntu 22.04, 24.04 and 25.04, and use the `uv` package management system to provide the required python packages. If you don't have it installed yet, check here: [install uv](https://docs.astral.sh/uv/getting-started/installation/).

Install python bindings for the gstreamer libraries, and the MQTT broker:

```
sudo apt install libgirepository1.0-dev libgirepository2.0-dev python3-gst-1.0 libcairo2-dev mosquitto git

uv sync
```

After the installation, libcairo-dev and its dependencies can be removed:

```
sudo apt remove libcairo2-dev
sudo apt autoremove
```

- Download the VAD model and the Faster Whisper models (large-v3-turbo by default)
  (faster whisper from [Huggingface](https://huggingface.co/guillaumekln))

```commandline
download-models.sh <optional-list-of-whisper-model-sizes>
```

## After package installation

Maybe you have to adapt the pipeline in `local_de_config.yml`, or the language, the current default expects a ReSpeaker as default PulseAudio device. For the ReSpeaker, use the multichannel, not the analog-stereo.monoto device! You can check your local audio device configuration with

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
./run_whisper.sh local_de_config.yml
```

The ASR result will be send to the `whisperasr/asrresult/<lang>` MQTT topic.

# Docker ASR server

To build a server docker image from this project, use the

    ./build_docker.sh

shell script. This will *not* include the ASR models, which have to be provided from the outside, as well as the configuration file. In the configuration file, the used model can be given, which allows to use different models easily, and also different containers with different models.

To run the dockerized version, use

    ./run_docker.sh <your_config.yml>

Be aware that this requires an MQTT broker that is running and accessible on the host system. The mosquitto service on a Linux system, for example, by default only accepts connections from localhost. To change that behaviour, you can create a file `10-externalconn.conf` in the directory `/etc/mosquitto/conf.d` with the following content:

    listener 1883 0.0.0.0
