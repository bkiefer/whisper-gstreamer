FROM ubuntu:25.04

ENV TZ=Europe/Berlin
RUN apt-get -q -qq update && apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    --no-install-recommends \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-alsa \
    gstreamer1.0-pulseaudio \
    gstreamer1.0-plugins-base-apps \
    ffmpeg \
    python3-pip \
    python3-cairo \
    python3-gst-1.0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app

RUN pip3 install --break-system-packages -r /app/requirements.txt
RUN rm -rf /root/.cache/pip
COPY src /app/src
COPY run_whisper.sh /app

CMD ["/bin/bash", "-c", "./run_whisper.sh -c config.cfg"]
