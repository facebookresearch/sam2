ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
ARG MODEL_SIZE=base_plus

FROM ${BASE_IMAGE}

# Gunicorn environment variables
ENV GUNICORN_WORKERS=1
ENV GUNICORN_THREADS=2
ENV GUNICORN_PORT=5000

# SAM 2 environment variables
ENV APP_ROOT=/opt/sam2
ENV PYTHONUNBUFFERED=1
ENV SAM2_BUILD_CUDA=0
ENV MODEL_SIZE=${MODEL_SIZE}

# Install system requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavutil-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    pkg-config \
    build-essential \
    libffi-dev

COPY setup.py .
COPY README.md .

RUN pip install --upgrade pip setuptools
RUN pip install -e ".[interactive-demo]"

# https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/issues/69#issuecomment-1826764707
RUN rm /opt/conda/bin/ffmpeg && ln -s /bin/ffmpeg /opt/conda/bin/ffmpeg

# Make app directory. This directory will host all files required for the
# backend and SAM 2 inference files.
RUN mkdir ${APP_ROOT}

# Copy backend server files
COPY demo/backend/server ${APP_ROOT}/server

# Copy SAM 2 inference files
COPY sam2 ${APP_ROOT}/server/sam2

# Download SAM 2.1 checkpoints
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_tiny.pt
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_small.pt
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_base_plus.pt
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_large.pt

WORKDIR ${APP_ROOT}/server

# https://pythonspeed.com/articles/gunicorn-in-docker/
CMD gunicorn --worker-tmp-dir /dev/shm \
    --worker-class gthread app:app \
    --log-level info \
    --access-logfile /dev/stdout \
    --log-file /dev/stderr \
    --workers ${GUNICORN_WORKERS} \
    --threads ${GUNICORN_THREADS} \
    --bind 0.0.0.0:${GUNICORN_PORT} \
    --timeout 60
