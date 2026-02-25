FROM mypy:3.11

WORKDIR /app
COPY src /app/src
COPY pyproject.toml run_whisper.sh /app
RUN uv sync
