#!/bin/bash

BASE_DIR=$(pwd)/${1:-som}

if [[ ! -d $BASE_DIR ]]
then
    SCRIPT_DIR=$(dirname "$(realpath $0)")
    echo < $SCRIPT_DIR/usage.txt
fi

echo "Mounting base directory $BASE_DIR into the container..."

# Run the container, exposing the port
docker run \
    --ipc=host \
    --gpus all \
    -p 8888:8888 \
    --mount type=bind,source="$BASE_DIR",target=/proj/som \
    fastai-jupyter &
sleep 4 && xdg-open http://localhost:8888/lab