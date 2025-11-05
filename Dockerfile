FROM ubuntu:25.04

ENV TZ=Europe/Berlin
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -q -qq update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends --fix-missing \
    git curl ca-certificates \
    gcc cmake pkg-config libcairo2-dev libgirepository-2.0-dev python3-dev \
    libgstreamer1.0-dev \
    gstreamer1.0-pulseaudio gstreamer1.0-alsa \
    python3-gst-1.0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Download the latest uv installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh
# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh
# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app
RUN git clone https://github.com/bkiefer/whisper-gstreamer /app
RUN uv sync

CMD ["uv", "run", "./run_whisper.sh", "-c", "config.cfg"]
