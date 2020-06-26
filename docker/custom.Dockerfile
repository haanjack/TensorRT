# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ubuntu:18.04

# Add NVIDIA repositories
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates \
    software-properties-common && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl && \
    rm -rf /var/lib/apt/lists/*

# CUDA versions
ENV CUDA_VERSION 10.2.89
ENV CUDA_PKG_VERSION 10-2=$CUDA_VERSION-1
ENV CUDNN_VERSION 8.0.0.180
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
ENV NCCL_VERSION 2.5.6
ENV TRT_VERSION 7.1.3-1

# Install requried libraries
# RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
        libcurl4-openssl-dev \
        wget \
        zlib1g-dev \
        git \
        pkg-config \
        python3 \
        python3-pip \
        python3-dev \
        python3-setuptools \
        python3-wheel \
        libc6 \
        sudo \
        ssh \
        pbzip2 \
        pv \
        bzip2 \
        unzip && \
    rm -rf /var/lib/apt/lists/*

RUN cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION \
        cuda-compat-10-2 \
        cuda-libraries-$CUDA_PKG_VERSION \
        cuda-nvtx-$CUDA_PKG_VERSION \
        libcublas10=10.2.2.89-1 \
        libnccl2=$NCCL_VERSION-1+cuda10.2 \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        libnccl-dev=$NCCL_VERSION-1+cuda10.2 \
        libcublas-dev=10.2.2.89-1 \
        libcudnn8=$CUDNN_VERSION-1+cuda10.2 \
        libcudnn8-dev=$CUDNN_VERSION-1+cuda10.2 \
        libnvinfer7=$TRT_VERSION+cuda10.2 \
        libnvinfer-dev=$TRT_VERSION+cuda10.2 \
        libnvinfer-plugin7=$TRT_VERSION+cuda10.2 \
        libnvinfer-plugin-dev=$TRT_VERSION+cuda10.2 \
        libnvparsers7=$TRT_VERSION+cuda10.2 \
        libnvparsers-dev=$TRT_VERSION+cuda10.2 \
        libnvonnxparsers7=$TRT_VERSION+cuda10.2 \
        libnvonnxparsers-dev=$TRT_VERSION+cuda10.2 \
        python3-libnvinfer=$TRT_VERSION+cuda10.2 \
        python3-libnvinfer-dev=$TRT_VERSION+cuda10.2 \
        && \
    apt-mark hold libcudnn8 && \
    apt-mark hold libnccl2 && \
    ln -s cuda-10.2 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.2 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419"

# Install Cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh && \
    chmod +x cmake-3.14.4-Linux-x86_64.sh && \
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.14.4-Linux-x86_64.sh
