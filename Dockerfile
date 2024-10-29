# Use an NVIDIA CUDA image as the base
# Thanks to https://github.com/peasant98/SAM2-Docker for the source

FROM nvidia/cuda:12.4.1-devel-ubuntu20.04

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="${PATH}:/home/user/.local/bin"

# We love UTF!
ENV LANG=C.UTF-8

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Set the nvidia container runtime environment variables
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME="/usr/local/cuda"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX 8.9"

# Install some handy tools. Even Guvcview for webcam support!
RUN set -x \
	&& apt-get update --fix-missing \
	&& apt-get install -y apt-transport-https ca-certificates \
	&& apt-get install -y git vim tmux nano htop sudo curl wget gnupg2 \
	&& apt-get install -y bash-completion \
	&& apt-get install -y guvcview \
	&& rm -rf /var/lib/apt/lists/* \
	&& useradd -ms /bin/bash user \
	&& echo "user:user" | chpasswd && adduser user sudo \
	&& echo "user ALL=(ALL) NOPASSWD: ALL " >> /etc/sudoers

RUN set -x \
    && apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN set -x \
    && apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-venv python3.11-dev \
    && apt-get install -y python3.11-tk

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /home/user

COPY ./segment-anything-2 ./segment-anything-2
RUN cd segment-anything-2 && \
    python3 -m pip install -e . -v && \
    python3 -m pip install -e ".[demo]"
RUN cd segment-anything-2/checkpoints && ls -la && /bin/bash ./download_ckpts.sh

RUN python3 -m pip install supervision

RUN usermod -aG dialout user
USER user
STOPSIGNAL SIGTERM

# COPY ./script /home/user/script

CMD ["sudo", "service", "ssh", "start", "&&", "/bin/bash"]

# docker build -t tchataing/sam2 -f ./Dockerfile_sam .
# docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix  -e DISPLAY=$DISPLAY -p 8888:8888 --gpus all tchataing/sam2 bash
# docker run -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix  -e DISPLAY=$DISPLAY -p 8888:8888 --gpus all tchataing/sam2 bash
# docker run -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix -v $PWD/input/8100_dir:/home/user/input -v $PWD/output:/home/user/output -e DISPLAY=$DISPLAY -p 8888:8888 --gpus all tchataing/sam2 bash

# sudo jupyter-lab --no-browser --ip 0.0.0.0 --allow-root


# for singularity singularity pull docker://tchataing/alphapose
