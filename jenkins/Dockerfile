FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      'python3.11' \
      'python3.11-venv' \
      ffmpeg \
      git \
      && \
    apt-get autoremove --purge -y && \
    apt-get autoclean -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*
