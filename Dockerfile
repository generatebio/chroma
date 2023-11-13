FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh
RUN /opt/conda/bin/conda create --name chroma python=3.9.7
WORKDIR /workspace
COPY . .
RUN /opt/conda/envs/chroma/bin/pip install .
ENV PATH /opt/conda/envs/chroma/bin:$PATH
