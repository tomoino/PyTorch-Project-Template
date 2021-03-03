#!/bin/sh
IMAGE_NAME=my_pytorch
CONTAINER_NAME=pytorch_project
docker run \
  -dit \
  --gpus all \
  -v $PWD:/workspace \
  -p 8888:8888 \
  --name $CONTAINER_NAME\
  --rm \
  --shm-size=2g \
  $IMAGE_NAME