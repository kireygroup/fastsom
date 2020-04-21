#!/bin/bash

BASE_DIR=$(pwd)/

# Reads credentials for SSH access
input="$BASE_DIR/bin/ssh-credentials.conf"
while read -r SSH_USER SSH_PASS; do
    echo ''
done < "$input"

# Builds the root folder Dockerfile
docker build $BASE_DIR -t fastsom --build-arg SSH_USER=$SSH_USER --build-arg SSH_PASS=$SSH_PASS