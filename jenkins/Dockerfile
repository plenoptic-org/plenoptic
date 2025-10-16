FROM pytorch/pytorch:2.9.0-cuda12.6-cudnn9-runtime

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      'python3.11' \
      'python3.11-venv' \
      git \
      curl \
      jq "ffmpeg" \
      pandoc \
      make \
      && \
    apt-get autoremove --purge -y && \
    apt-get autoclean -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*
