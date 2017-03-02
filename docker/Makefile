help:
	@cat Makefile

DATA?="${HOME}/Data"
GPU?=0
DOCKER_FILE_GPU=Dockerfile-gpu
DOCKER_GPU=GPU=$(GPU) nvidia-docker
BACKEND=tensorflow
TEST=tests/
SRC=$(shell dirname `pwd`)
CUDA=/usr/local/cuda
LD_LIBRARY=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64

build-gpu:
	docker build -t edward --build-arg python_version=3.5 -f $(DOCKER_FILE_GPU) .

bash-gpu: build-gpu
	$(DOCKER_GPU) run -it -v $(SRC):/src -v $(DATA):/data --env CUDA_HOME=${CUDA} \
	                                                  --env LD_LIBRARY_PATH=${LD_LIBRARY} \
	                                                  edward bash

ipython-gpu: build-gpu
	$(DOCKER_GPU) run -it -v $(SRC):/src -v $(DATA):/data --env CUDA_HOME=${CUDA} \
	                                                  --env LD_LIBRARY_PATH=${LD_LIBRARY} \
	                                                  edward ipython

notebook-gpu: build-gpu
	$(DOCKER_GPU) run -it -v $(SRC):/src -v $(DATA):/data --net=host --env CUDA_HOME=${CUDA} \
	                                                  --env LD_LIBRARY_PATH=${LD_LIBRARY} \
	                                                  edward

test-gpu: build-gpu
	$(DOCKER_GPU) run -it -v $(SRC):/src -v $(DATA):/data --env CUDA_HOME=${CUDA} \
	                                                  --env LD_LIBRARY_PATH=${LD_LIBRARY} \
	                                                  edward py.test $(TEST)

