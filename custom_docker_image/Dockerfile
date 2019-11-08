FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
RUN  apt-get update \ 
  && apt-get install -y wget git libsm6 libxext6 libxrender-dev libgtk2.0-dev \
  && rm -rf /var/lib/apt/lists/*
RUN apt-get update\
    && apt-get upgrade -y \
    && apt-get install bzip2 

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /miniconda
RUN /bin/bash -c "source ~/.bashrc"

RUN conda init
RUN /bin/bash -c "source ~/.bashrc"

RUN conda remove -c anaconda pyyaml -y

COPY requirements.txt /tmp
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
        && rm /tmp/requirements.txt

RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex
RUN pip install -v --no-cache-dir apex/

ENTRYPOINT ["python"]
