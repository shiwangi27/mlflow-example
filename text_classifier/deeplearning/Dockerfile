FROM pytorch/pytorch:latest
MAINTAINER "Shiwangi Singh"

#RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext

RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      unzip \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget && \
    rm -rf /var/lib/apt/lists/*

ARG python_version=3.6

RUN conda install -y python=${python_version} && \
    pip install --upgrade pip && \
    pip install \
      sklearn_pandas \
      tensorflow \
      nltk \
      cntk && \
    conda install \
      bcolz \
      h5py \
      matplotlib \
      mkl \
      nose \
      notebook \
      Pillow \
      pandas \
      pydot \
      pygpu \
      pyyaml \
      scikit-learn \
      six && \
    git clone https://github.com/keras-team/keras.git /src && pip install -e /src[tests] && \
    pip install git+https://github.com/keras-team/keras.git && \
    conda clean -yt

$WORK_DIR=/opt/transformers

RUN pip install transformers

RUN pip install -r requirements.txt


WORKDIR /workspace

