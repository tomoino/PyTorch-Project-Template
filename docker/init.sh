#!/bin/sh
#
# Process to run after the docker container is started.

. docker/env.sh
echo "python train.py -sc install=fish | source" >> ~/dotfiles/fish/config.fish
echo "\n" >> ~/dotfiles/fish/alias.fish
echo "alias jupyterlab 'jupyter lab --port ${JUPYTER_CONTAINER_PORT} --ip=0.0.0.0 --allow-root'" >> ~/dotfiles/fish/alias.fish
mlflow ui --host 0.0.0.0 -p $MLFLOW_CONTAINER_PORT 