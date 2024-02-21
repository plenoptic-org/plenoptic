FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3.11-numpy \
      python3.11-pip \
      python3.11-pytest \
      python3.11-pytest-cov \
      python3.11-venv \
      python3.11-yaml \
      ffmpeg \
      git \
      && \
    apt-get autoremove --purge -y && \
    apt-get autoclean -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*
