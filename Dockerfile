FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Set environment variables
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10.13

# Install python
# https://stackoverflow.com/questions/70866415/how-to-install-python-specific-version-on-docker
RUN apt update -y && apt upgrade -y && \
    apt-get install -y wget build-essential checkinstall  libreadline-gplv2-dev  libncursesw5-dev  libssl-dev  libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev && \
    cd /usr/src && \
    wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz && \
    tar xzf Python-3.10.13.tgz && \
    cd Python-3.10.13 && \
    ./configure --enable-optimizations && \
    make altinstall
RUN apt update -y && apt install -y ffmpeg
RUN pip3.10 install poetry

# RUN apt-get -y update \
#     && apt-get install -y software-properties-common \
#     && apt-get -y update \
#     && add-apt-repository universe
# RUN apt-get -y update
# RUN apt-get -y install python3.10
# RUN apt-get -y install python3-pip

# Copy over test code, install dependencies, and download model
RUN mkdir /root/whisper-benchmark
COPY . /root/whisper-benchmark
WORKDIR /root/whisper-benchmark
RUN poetry install
RUN poetry run python download_whisper_large_v2.py

CMD ["poetry", "run", "python", "test.py"]