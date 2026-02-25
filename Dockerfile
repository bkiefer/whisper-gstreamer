FROM mypy:3.11

WORKDIR /app
RUN git init
RUN git remote add origin https://github.com/bkiefer/whisper-gstreamer
RUN git fetch
RUN git checkout -t origin/master
RUN uv sync
