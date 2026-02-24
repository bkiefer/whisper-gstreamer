FROM mypy:3.11

WORKDIR /app
RUN git clone https://github.com/bkiefer/whisper-gstreamer /app
RUN uv sync
