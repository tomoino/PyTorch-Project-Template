#!/bin/sh
#
# Process to run after the docker container is started.

. docker/env.sh
echo -e "\npython train.py -sc install=fish | source" >> ~/dotfiles/fish/config.fish
mlflow ui --host 0.0.0.0 -p $CONTAINER_PORT 