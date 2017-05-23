# Using Edward via Docker

This directory contains `Dockerfile` to make it easy to get up and running with
Edward via [Docker](http://www.docker.com/).

## Installing Docker

General installation instructions are
[on the Docker site](https://docs.docker.com/installation/), but we give some
quick links here:

* [OSX](https://docs.docker.com/installation/mac/): [docker toolbox](https://www.docker.com/toolbox)
* [ubuntu](https://docs.docker.com/installation/ubuntulinux/)

## Installing NVIDIA Docker (GPU Environment)

General installation instructions are
[on the NVIDIA Docker site](https://github.com/NVIDIA/nvidia-docker)

## Running the container

We are using `Makefile` to simplify docker commands within make commands.

### CPU environment

Build the container and start a jupyter notebook

    $ make notebook

Build the container and start an iPython shell

    $ make ipython

Build the container and start a bash

    $ make bash

Build the container and start a test

    $ make test

### GPU environment

Build the container and start a jupyter notebook

    $ make notebook-gpu

Build the container and start an iPython shell

    $ make ipython-gpu

Build the container and start a bash

    $ make bash-gpu

Build the container and start a test

    $ make test-gpu

For GPU support install NVidia drivers (ideally latest) and
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker). Run using

    $ make notebook-gpu GPU=0 # or [ipython, bash]

Mount a volume for external data sets

    $ make DATA=~/mydata

Prints all make tasks

    $ make help


Note: If you would have a problem running nvidia-docker you may try the old way
we have used. But it is not recommended. If you find a bug in the nvidia-docker report
it there please and try using the nvidia-docker as described above.

    $ export CUDA_SO=$(\ls /usr/lib/x86_64-linux-gnu/libcuda.* | xargs -I{} echo '-v {}:{}')
    $ export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
    $ docker run -it -p 8888:8888 $CUDA_SO $DEVICES gcr.io/tensorflow/tensorflow:latest-gpu
