#################################################################################################################
# Base Images
#################################################################################################################
FROM ubuntu:14.04

#################################################################################################################
#           ENV Setting
#################################################################################################################
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH
ENV LANG C.UTF-8

#################################################################################################################
#           Initial Setting
#################################################################################################################
RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    apt-get update && \
    apt-get install -y wget git libhdf5-dev g++ graphviz && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash /Miniconda3-latest-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh

#################################################################################################################
#           User Setting
#################################################################################################################
ENV NB_USER edward
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown edward $CONDA_DIR -R && \
    mkdir -p /src && \
    chown edward /src

USER edward

#################################################################################################################
#           Python Setting
#################################################################################################################
# Python
ARG python_version=3.5.3-0
ARG python_qt_version=4
RUN conda install -y python=${python_version} && \
    pip install observations numpy six tensorflow keras prettytensor && \
    pip install ipdb pytest pytest-cov python-coveralls coverage==3.7.1 pytest-xdist pep8 pytest-pep8 pydot_ng && \
    conda install Pillow scikit-learn matplotlib notebook pandas seaborn pyyaml h5py && \
    conda install -y pyqt=${python_qt_version} && \
    pip install edward && \
    conda clean -yt

ENV PYTHONPATH='/src/:$PYTHONPATH'

#################################################################################################################
#           WORK Jupyter
#################################################################################################################
WORKDIR /src

EXPOSE 8888

CMD jupyter notebook --port=8888 --ip=0.0.0.0
