# choose the right cuda image based on your gpu
FROM nvidia/cuda:11.6.0-runtime-ubuntu20.04

# install python
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London
RUN apt update && \
    apt install -y git && \
    apt install --no-install-recommends -y python3.8 python3-pip python3-setuptools python3-distutils python3-tk && \
    apt install -y wget && \
    apt install -y unzip && \
    apt clean && rm -rf /var/lib/apt/lists/*

# default workdir
WORKDIR /app

# install python dependencies
COPY ./requirements.txt ./requirements.txt
RUN python3.8 -m pip install -r requirements.txt
RUN python3.8 -m pip install git+https://github.com/oxwhirl/smac.git

# install pytorch, change version also here if you're changing the cuda image
RUN python3.8 -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# install StarCraft2
COPY ./install_sc2.sh ./install_sc2.sh
RUN chmod +x install_sc2.sh
RUN ./install_sc2.sh