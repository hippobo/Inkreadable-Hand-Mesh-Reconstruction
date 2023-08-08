# Use an official Python runtime as a parent image
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000
# Set environment variables
ENV FLASK_APP=flask_app/app.py
ENV PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install necessary packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    software-properties-common gcc git wget libgl1-mesa-glx libglu1-mesa libsm6 libxext6 libxrender-dev

# Install Python 3.7 and pip
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.7 python3.7-venv python3.7-distutils python3.7-dev python3-pip 

# Set Python3.7 as the default version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

RUN python3 setup.py build develop

# Install pip packages
RUN pip3 install --upgrade pip && \
    pip3 install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip3 install -r requirements.txt

# Clone and install apex
RUN git clone https://github.com/ptrblck/apex.git && \
    cd apex && \
    python3 setup.py install --cuda_ext --cpp_ext

# Install manopth
RUN pip3 install ./manopth/.

# Download Blender
RUN bash scripts/download_blender_linux.sh

RUN bash scripts/create_dirs.sh

# Create a non-root user and add to video group
RUN useradd -ms /bin/bash myuser && usermod -a -G video myuser

# Change ownership of /app directory to myuser
RUN chown -R myuser:myuser /app

# Switch to the non-root user
USER myuser

# Run the application
CMD ["flask", "run"]
