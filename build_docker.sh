cd basedocker; build_docker.sh; cd ..

. utils.sh
docker build -f Dockerfile -t "$(getimage)" .
