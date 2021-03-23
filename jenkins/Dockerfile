FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3-pip \
      python3-pytest \
      python3-pytest-cov \
      python3-venv \
      ffmpeg \
      && \
    apt-get autoremove --purge -y && \
    apt-get autoclean -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

RUN pip3 install \
        "numpy>=1.1" \
        "torch<1.8" \
        "pyrtools>=1.0.0" \
        "scipy>=1.0" \
        "matplotlib>=3.1" \
        "torchvision>=0.3" \
        "tqdm>=4.29" \
        "requests>=2.21" \
        "imageio>=2.5" \
        "pytest" \
        "scikit-image>=0.15.0" \
        "dill"
