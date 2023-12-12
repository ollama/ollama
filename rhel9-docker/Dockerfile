FROM registry.access.redhat.com/ubi9/ubi:latest

# Install Nvidia Driver and CUDA Toolkit
ENV NVARCH x86_64
ENV NVIDIA_REQUIRE_CUDA "cuda>=12.3 brand=tesla,driver>=470,driver<471 ... driver<536"
ENV NV_CUDA_CUDART_VERSION 12.3.52-1
COPY cuda.repo-x86_64 /etc/yum.repos.d/cuda.repo

LABEL maintainer "NVIDIA CORPORATION <sw-cuda-installer@nvidia.com>"

RUN NVIDIA_GPGKEY_SUM=d0664fbbdb8c32356d45de36c5984617217b2d0bef41b93ccecd326ba3b80c87 && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel9/${NVARCH}/D42D0685.pub | sed '/^Version/d' > /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA && \
    echo "$NVIDIA_GPGKEY_SUM  /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA" | sha256sum -c --strict -

ENV CUDA_VERSION 12.3.0

RUN yum upgrade -y && yum install -y \
    cuda-cudart-12-3-${NV_CUDA_CUDART_VERSION} \
    cuda-compat-12-3 \
    && yum clean all \
    && rm -rf /var/cache/yum/*

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

# Install wget
RUN yum install -y wget

# Install Ollama
RUN curl -L https://ollama.ai/download/ollama-linux-amd64 -o /usr/bin/ollama && \
    chmod +x /usr/bin/ollama

# Create a non-root user named 'ollama'
RUN useradd -m -u 1000 -U -s /bin/bash ollama

# Set the working directory to the user's home directory
WORKDIR /home/ollama

# Switch to the non-root user
USER ollama

# Create a directory for the conda installation
RUN mkdir -p ./miniconda3

# Install Miniconda3 with Python 3.11
RUN mkdir -p ./miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -u -p ./miniconda3 && \
    rm -rf miniconda.sh

# Copy the requirements file and entrypoint script to the container
COPY requirements.txt .
COPY entrypoint.sh .
COPY config.yaml .

# Install requirements
RUN ./miniconda3/bin/pip install -r ./requirements.txt

# Set the path and library path environment for the container
ENV PATH /home/ollama:/usr/bin/ollama:/home/ollama/miniconda3/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:/home/ollama/miniconda3/lib:${LD_LIBRARY_PATH}

#Copy the Nvidia license file to the container
COPY NGC-DL-CONTAINER-LICENSE /

#Set Nvidia environment variables
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Set the entrypoint script as the container entrypoint
ENTRYPOINT [ "/bin/bash", "./entrypoint.sh" ]