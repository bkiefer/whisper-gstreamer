docker images 2>&1 | grep -q mypy:3.11 ||
    docker build -f Dockerfile_mypy3_11 -t mypy:3.11 .

. utils.sh
docker build -f Dockerfile -t "$(getimage)" .
