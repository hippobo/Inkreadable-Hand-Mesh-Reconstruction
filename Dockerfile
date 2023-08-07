FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    software-properties-common gcc git wget

RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.7 python3.7-venv python3.7-distutils python3.7-dev python3-pip 

# Set Python3.7 as the default version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

RUN python3 setup.py build develop

ARG PIP_NO_CACHE_DIR=1
# Set pip default timeout
ENV PIP_DEFAULT_TIMEOUT=100

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglu1-mesa



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

# Create models directory
RUN mkdir -p models

# Download pre-trained models
RUN bash scripts/download_models.sh

# Expose port for the server
EXPOSE 8888

# Expose port for the Flask web server
EXPOSE 5000


ENV FLASK_APP=app.py

RUN pip3 freeze


# Run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
