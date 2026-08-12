# =============================================================================
# ISIC MILK10k submission container.
#
# Contract (https://github.com/ImageMarkup/isic-algorithm-example):
#   - score every JPEG under /images
#   - write CSV with headers to stdout
#
# Build from the REPOSITORY ROOT so src/ and docker/ are both in context:
#   docker build -f docker/Dockerfile -t <dockerhub-user>/milk10k:latest .
#
# Test:
#   docker run --rm -v /path/to/MILK10k_Test_Input:/images \
#       <dockerhub-user>/milk10k:latest > submission.csv
#
# CPU-only base image: the challenge does not guarantee a GPU, and algorithm.py
# falls back to CPU automatically. Swap in a CUDA base if GPU inference is wanted.
# =============================================================================
FROM python:3.11-slim

# libgl/libglib are needed by opencv; no recommends keeps the layer small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/algorithm

# CPU-only torch wheels: the default CUDA build would add several GB for nothing.
RUN pip install --no-cache-dir \
        torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        timm==1.0.28 \
        numpy \
        opencv-python-headless \
        scikit-learn \
        scipy \
        pyyaml

# Model code, then the entrypoint, then the weights. Weights last so that editing
# code does not invalidate the ~820 MB layer.
COPY src/ ./src/
COPY docker/algorithm.py ./algorithm.py
COPY weights_bundle/ ./weights_bundle/

# timm must never reach the network at run time: the bundle carries every weight.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4

ENTRYPOINT ["python", "algorithm.py"]
