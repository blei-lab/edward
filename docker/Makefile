help:
	@cat Makefile

DATA?="${HOME}/Data"
GPU?=0
DOCKER_FILE_GPU=Dockerfile-gpu
DOCKER=GPU=$(GPU) nvidia-docker
BACKEND=tensorflow
TEST=tests/
SRC=$(shell dirname `pwd`)

build:
	docker build -t edward --build-arg python_version=3.5 -f $(DOCKER_FILE_GPU) .

bash: build
	$(DOCKER) run -it -v $(SRC):/src -v $(DATA):/data --env EDWARD_BACKEND=$(BACKEND) edward bash

ipython: build
	$(DOCKER) run -it -v $(SRC):/src -v $(DATA):/data --env EDWARD_BACKEND=$(BACKEND) edward ipython

notebook: build
	$(DOCKER) run -it -v $(SRC):/src -v $(DATA):/data --net=host --env EDWARD_BACKEND=$(BACKEND) edward

test: build
	$(DOCKER) run -it -v $(SRC):/src -v $(DATA):/data --env CUDA_HOME=/usr/local/cuda \
	                                                  --env LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
	                                                  edward py.test $(TEST)

