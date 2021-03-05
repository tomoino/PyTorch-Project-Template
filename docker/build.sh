#!/bin/sh
IMAGE_NAME=my_pytorch
docker build \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  -f docker/Dockerfile \
  -t $IMAGE_NAME \
  --force-rm=true \
  .