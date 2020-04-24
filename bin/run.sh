#!/bin/bash

CODE_DIR=$(pwd)/${1:-fastsom}
NBS_DIR=$(pwd)/${1:-nbs}

echo "Mounting directories $CODE_DIR and $NBS_DIR into the container..."

#  Fork a process to open a new browser tab
sleep 4 && xdg-open http://localhost:8888/lab &

# Run the container, exposing the port
docker run \
    --rm \
    --ipc=host \
    --gpus all \
    -p 8888:8888 -p 8787:8787 -p 8786:8786 -p 9091:22 \
    --mount type=bind,source="$CODE_DIR",target=/proj/fastsom/fastsom \
    --mount type=bind,source="$NBS_DIR",target=/proj/fastsom/nbs \
    fastsom
