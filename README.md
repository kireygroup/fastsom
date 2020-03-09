# Self - Organizing Maps

## Get Started

### Prerequisites

To run this project you need to install Docker or Nvidia-Docker.

### Build the image

An utility script can be found in `bin/build.sh`:

```bash
./bin/build.sh
```

### Run the image

A run script is available:

```bash
./bin/run.sh
```

This will mount the `/som` directory inside the container, allowing for code changes to be automatically replicated.

Note: if you plan on using Nvidia-Docker, you should use one of the images available on the Nvidia Container Repository.

The container will start a new Jupyter Notebook server on port 8888. Jupyter Lab is also available.

Note that the som folder will be mounted inside the container, so any change you make to the source files or notebooks will be replicated on both systems.

## Project Structure

```text
som/
    nbs/
    som/
bin/
```

## Developing inside the container

With Visual Studio Code and PyCharm, it is possible to use the container Python interpreter for development.

An SSH server has been configured to allow connection via PyCharm's remote interpreter feature.

In Visual Studio Code, this can be done via the [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
