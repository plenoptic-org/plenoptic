FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      cmake \
      git \
      intel-mkl \
      python3-numpy \
      python3-pip \
      python3-pytest \
      python3-pytest-cov \
      python3-venv \
      python3-yaml \
      ffmpeg \
      && \
    apt-get autoremove --purge -y && \
    apt-get autoclean -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

RUN git clone --recursive --depth 1 -b v1.7.1 https://github.com/pytorch/pytorch /tmp/pytorch && \
    cd /tmp/pytorch && \
    CMAKE_INCLUDE_PATH=/usr/include/mkl TORCH_CUDA_ARCH_LIST="3.5" python3 setup.py install && \
    rm -rf /tmp/pytorch

COPY requirements.txt /tmp/
RUN pip3 install --no-binary=torch -r /tmp/requirements.txt
