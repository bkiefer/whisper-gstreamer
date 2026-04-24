docker build -f Dockerfile_mypy3_11 -t mypy:3.11 .

version=`grep version pyproject.toml | sed 's/.*"\([^"]*\)".*/\1/'`
#echo $version
docker build -f Dockerfile -t whisper_asr:"$version" .
