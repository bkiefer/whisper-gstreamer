cd basedocker && ./build_docker.sh && cd .. || exit 1

. utils.sh
docker build -f Dockerfile -t "$(getimage)" .
