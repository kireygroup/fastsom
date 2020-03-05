#!/bin/bash

BASE_DIR=$(pwd)/

# Builds the root folder Dockerfile
docker build -t fastai-jupyter $BASE_DIR