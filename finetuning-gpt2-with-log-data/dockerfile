FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y python3-dev htop wget 

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && sh -f Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

COPY . src/

RUN conda create -y -n ml python=3.8

RUN /bin/bash -c "cd src \
    && source activate ml \
    && pip install -r requirements.txt"
