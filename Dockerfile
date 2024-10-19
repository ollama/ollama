ARG GOLANG_VERSION=1.22.5
ARG CMAKE_VERSION=3.22.1
ARG CUDA_VERSION_11=11.7.1
ARG CUDA_V11_ARCHITECTURES="50;52;53;60;61;62;70;72;75;80;86"
ARG CUDA_VERSION_12=12.4.0
ARG CUDA_V12_ARCHITECTURES="60;61;62;70;72;75;80;86;87;89;90;90a"
ARG ROCM_VERSION=6.1.2

# Copy the minimal context we need to run the generate scripts
FROM scratch AS llm-code
COPY .git .git
COPY .gitmodules .gitmodules
COPY llm llm

FROM --platform=linux/amd64 nvidia/cuda:$CUDA_VERSION_11-devel-rockylinux8 AS cuda-11-build-amd64
ARG CMAKE_VERSION
COPY ./scripts/rh_linux_deps.sh /
RUN CMAKE_VERSION=${CMAKE_VERSION} sh /rh_linux_deps.sh
ENV PATH=/opt/rh/devtoolset-11/root/usr/bin:$PATH
COPY --from=llm-code / /go/src/github.com/ollama/ollama/
WORKDIR /go/src/github.com/ollama/ollama/llm/generate
ARG CGO_CFLAGS
ARG CUDA_V11_ARCHITECTURES
ENV GOARCH=amd64
RUN --mount=type=cache,target=/root/.ccache \
    OLLAMA_SKIP_STATIC_GENERATE=1 \
    OLLAMA_SKIP_CPU_GENERATE=1 \
    CMAKE_CUDA_ARCHITECTURES="${CUDA_V11_ARCHITECTURES}" \
    CUDA_VARIANT="_v11" \
    bash gen_linux.sh

FROM --platform=linux/amd64 nvidia/cuda:$CUDA_VERSION_12-devel-rockylinux8 AS cuda-12-build-amd64
ARG CMAKE_VERSION
COPY ./scripts/rh_linux_deps.sh /
RUN CMAKE_VERSION=${CMAKE_VERSION} sh /rh_linux_deps.sh
ENV PATH=/opt/rh/devtoolset-11/root/usr/bin:$PATH
COPY --from=llm-code / /go/src/github.com/ollama/ollama/
WORKDIR /go/src/github.com/ollama/ollama/llm/generate
ARG CGO_CFLAGS
ARG CUDA_V12_ARCHITECTURES
ENV GOARCH=amd64
RUN --mount=type=cache,target=/root/.ccache \
    OLLAMA_SKIP_STATIC_GENERATE=1 \
    OLLAMA_SKIP_CPU_GENERATE=1 \
    CMAKE_CUDA_ARCHITECTURES="${CUDA_V12_ARCHITECTURES}" \
    CUDA_VARIANT="_v12" \
    OLLAMA_CUSTOM_CUDA_DEFS="-DGGML_CUDA_USE_GRAPHS=on" \
    bash gen_linux.sh

FROM --platform=linux/arm64 nvidia/cuda:$CUDA_VERSION_11-devel-rockylinux8 AS cuda-11-build-runner-arm64
ARG CMAKE_VERSION
COPY ./scripts/rh_linux_deps.sh /
RUN CMAKE_VERSION=${CMAKE_VERSION} sh /rh_linux_deps.sh
ENV PATH=/opt/rh/gcc-toolset-11/root/usr/bin:$PATH
COPY --from=llm-code / /go/src/github.com/ollama/ollama/
WORKDIR /go/src/github.com/ollama/ollama/llm/generate
ARG CGO_CFLAGS
ARG CUDA_V11_ARCHITECTURES
ENV GOARCH=arm64
RUN OLLAMA_SKIP_STATIC_GENERATE=1 \
    OLLAMA_SKIP_CPU_GENERATE=1 \
    CMAKE_CUDA_ARCHITECTURES="${CUDA_V11_ARCHITECTURES}" \
    CUDA_VARIANT="_v11" \
    bash gen_linux.sh

FROM --platform=linux/arm64 nvidia/cuda:$CUDA_VERSION_12-devel-rockylinux8 AS cuda-12-build-runner-arm64
ARG CMAKE_VERSION
COPY ./scripts/rh_linux_deps.sh /
RUN CMAKE_VERSION=${CMAKE_VERSION} sh /rh_linux_deps.sh
ENV PATH=/opt/rh/gcc-toolset-11/root/usr/bin:$PATH
COPY --from=llm-code / /go/src/github.com/ollama/ollama/
WORKDIR /go/src/github.com/ollama/ollama/llm/generate
ARG CGO_CFLAGS
ARG CUDA_V12_ARCHITECTURES
ENV GOARCH=arm64
RUN --mount=type=cache,target=/root/.ccache \
    OLLAMA_SKIP_STATIC_GENERATE=1 \
    OLLAMA_SKIP_CPU_GENERATE=1 \
    CMAKE_CUDA_ARCHITECTURES="${CUDA_V12_ARCHITECTURES}" \
    CUDA_VARIANT="_v12" \
    OLLAMA_CUSTOM_CUDA_DEFS="-DGGML_CUDA_USE_GRAPHS=on" \
    bash gen_linux.sh

FROM --platform=linux/amd64 rockylinux:8 as rocm-dev-rockylinux
# note this is an exact copy of the official rocm almalinux image at https://github.com/ROCm/ROCm-docker/blob/db86386c24eeb45f5d3ba73564b00cc66566e537/dev/Dockerfile-almalinux-8-complete
# with 2 changes:
#  - use gcc 11 instead of 9
#  - base on rockylinux instead of almalinux
ARG ROCM_VERSION=6.1.2
ARG AMDGPU_VERSION=${ROCM_VERSION}

# Base
RUN yum -y install git java-1.8.0-openjdk python39; yum clean all

# Enable epel-release repositories
RUN dnf install -y 'dnf-command(config-manager)'
RUN dnf config-manager --set-enabled powertools
RUN dnf install -y epel-release

# Install required base build and packaging commands for ROCm
RUN yum -y install \
    ca-certificates \
    bc \
    bridge-utils \
    cmake \
    cmake3 \
    dkms \
    doxygen \
    dpkg \
    dpkg-dev \
    dpkg-perl \
    elfutils-libelf-devel \
    expect \
    file \
    python3-devel \
    python3-pip \
    gettext \
    gcc-c++ \
    libgcc \
    lzma \
    glibc.i686 \
    ncurses \
    ncurses-base \
    ncurses-libs \
    numactl-devel \
    numactl-libs \
    libssh \
    libunwind-devel \
    libunwind \
    llvm \
    llvm-libs \
    make \
    openssl \
    openssl-libs \
    openssh \
    openssh-clients \
    pciutils \
    pciutils-devel \
    pciutils-libs \
    perl \
    pkgconfig \
    qemu-kvm \
    re2c \
    kmod \
    rpm \
    rpm-build \
    subversion \
    wget

# Enable the epel repository for fakeroot
RUN yum install -y fakeroot
RUN yum clean all

# todo here we should prob go to 10 for consistency with the rest of everything. or maybe 11
# Install devtoolset 11
RUN yum install -y gcc-toolset-11
RUN yum install -y gcc-toolset-11-libatomic-devel gcc-toolset-11-elfutils-libelf-devel

# Install ROCm repo paths
RUN echo -e "[ROCm]\nname=ROCm\nbaseurl=https://repo.radeon.com/rocm/rhel8/$ROCM_VERSION/main\nenabled=1\ngpgcheck=0\npriority=50" >> /etc/yum.repos.d/rocm.repo
RUN echo -e "[amdgpu]\nname=amdgpu\nbaseurl=https://repo.radeon.com/amdgpu/$AMDGPU_VERSION/rhel/8.9/main/x86_64\nenabled=1\ngpgcheck=0\npriority=50" >> /etc/yum.repos.d/amdgpu.repo

# Install versioned ROCm packages eg. rocm-dev6.1.0 to avoid issues with "yum update" pulling really old rocm-dev packages from epel
# COPY scripts/install_versioned_rocm.sh install_versioned_rocm.sh
# RUN bash install_versioned_rocm.sh ${ROCM_VERSION}
# RUN rm install_versioned_rocm.sh

RUN yum install -y rocm-dev rocm-libs

# Set ENV to enable devtoolset9 by default
ENV PATH=/opt/rh/gcc-toolset-11/root/usr/bin:/opt/rocm/bin:${PATH:+:${PATH}}
ENV MANPATH=/opt/rh/gcc-toolset-11/root/usr/share/man:${MANPATH}
ENV INFOPATH=/opt/rh/gcc-toolset-11/root/usr/share/info:${INFOPATH:+:${INFOPATH}}
ENV PCP_DIR=/opt/rh/gcc-toolset-11/root
ENV PERL5LIB=/opt/rh/gcc-toolset-11/root/usr/lib64/perl5/vendor_perl
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib:/opt/rh/gcc-toolset-11/root/lib:/opt/rh/gcc-toolset-11/root/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# ENV PYTHONPATH=/opt/rh/gcc-toolset-11/root/

ENV LDFLAGS="-Wl,-rpath=/opt/rh/gcc-toolset-11/root/usr/lib64 -Wl,-rpath=/opt/rh/gcc-toolset-11/root/usr/lib"

FROM --platform=linux/amd64 rocm-dev-rockylinux AS rocm-build-amd64
ARG CMAKE_VERSION
COPY ./scripts/rh_linux_deps.sh /
RUN CMAKE_VERSION=${CMAKE_VERSION} sh /rh_linux_deps.sh
ENV PATH=/opt/rh/gcc-toolset-11/root/usr/bin:$PATH
ENV LIBRARY_PATH=/opt/amdgpu/lib64
COPY --from=llm-code / /go/src/github.com/ollama/ollama/
WORKDIR /go/src/github.com/ollama/ollama/llm/generate
ARG CGO_CFLAGS
ARG AMDGPU_TARGETS
ENV GOARCH=amd64
RUN --mount=type=cache,target=/root/.ccache \
    OLLAMA_SKIP_STATIC_GENERATE=1 OLLAMA_SKIP_CPU_GENERATE=1 bash gen_linux.sh
RUN mkdir -p ../../dist/linux-amd64-rocm/lib/ollama && \
    (cd /opt/rocm/lib && tar cf - rocblas/library) | (cd ../../dist/linux-amd64-rocm/lib/ollama && tar xf - )

FROM --platform=linux/amd64 rockylinux:8 AS cpu-builder-amd64
ARG CMAKE_VERSION
ARG GOLANG_VERSION
COPY ./scripts/rh_linux_deps.sh /
RUN CMAKE_VERSION=${CMAKE_VERSION} GOLANG_VERSION=${GOLANG_VERSION} sh /rh_linux_deps.sh
ENV PATH=/opt/rh/gcc-toolset-11/root/usr/bin:$PATH
COPY --from=llm-code / /go/src/github.com/ollama/ollama/
ARG OLLAMA_CUSTOM_CPU_DEFS
ARG CGO_CFLAGS
ENV GOARCH=amd64
WORKDIR /go/src/github.com/ollama/ollama/llm/generate

FROM --platform=linux/amd64 cpu-builder-amd64 AS cpu-build-amd64
RUN --mount=type=cache,target=/root/.ccache \
    OLLAMA_SKIP_STATIC_GENERATE=1 OLLAMA_CPU_TARGET="cpu" bash gen_linux.sh
FROM --platform=linux/amd64 cpu-builder-amd64 AS cpu_avx-build-amd64
RUN --mount=type=cache,target=/root/.ccache \
    OLLAMA_SKIP_STATIC_GENERATE=1 OLLAMA_CPU_TARGET="cpu_avx" bash gen_linux.sh
FROM --platform=linux/amd64 cpu-builder-amd64 AS cpu_avx2-build-amd64
RUN --mount=type=cache,target=/root/.ccache \
    OLLAMA_SKIP_STATIC_GENERATE=1 OLLAMA_CPU_TARGET="cpu_avx2" bash gen_linux.sh

FROM --platform=linux/arm64 rockylinux:8 AS cpu-builder-arm64
ARG CMAKE_VERSION
ARG GOLANG_VERSION
COPY ./scripts/rh_linux_deps.sh /
RUN CMAKE_VERSION=${CMAKE_VERSION} GOLANG_VERSION=${GOLANG_VERSION} sh /rh_linux_deps.sh
ENV PATH=/opt/rh/gcc-toolset-11/root/usr/bin:$PATH
COPY --from=llm-code / /go/src/github.com/ollama/ollama/
ARG OLLAMA_CUSTOM_CPU_DEFS
ARG CGO_CFLAGS
ENV GOARCH=arm64
WORKDIR /go/src/github.com/ollama/ollama/llm/generate

FROM --platform=linux/arm64 cpu-builder-arm64 AS cpu-build-arm64
RUN --mount=type=cache,target=/root/.ccache \
    OLLAMA_SKIP_STATIC_GENERATE=1 OLLAMA_CPU_TARGET="cpu" bash gen_linux.sh


# Intermediate stages used for ./scripts/build_linux.sh
FROM --platform=linux/amd64 cpu-build-amd64 AS build-amd64
ENV CGO_ENABLED=1
WORKDIR /go/src/github.com/ollama/ollama
COPY . .
COPY --from=cpu_avx-build-amd64 /go/src/github.com/ollama/ollama/build/ build/
COPY --from=cpu_avx2-build-amd64 /go/src/github.com/ollama/ollama/build/ build/
COPY --from=cuda-11-build-amd64 /go/src/github.com/ollama/ollama/dist/ dist/
COPY --from=cuda-11-build-amd64 /go/src/github.com/ollama/ollama/build/ build/
COPY --from=cuda-12-build-amd64 /go/src/github.com/ollama/ollama/dist/ dist/
COPY --from=cuda-12-build-amd64 /go/src/github.com/ollama/ollama/build/ build/
COPY --from=rocm-build-amd64 /go/src/github.com/ollama/ollama/dist/ dist/
COPY --from=rocm-build-amd64 /go/src/github.com/ollama/ollama/build/ build/
ARG GOFLAGS
ARG CGO_CFLAGS
RUN --mount=type=cache,target=/root/.ccache \
    go build -trimpath -o dist/linux-amd64/bin/ollama .
RUN cd dist/linux-$GOARCH && \
    tar --exclude runners -cf - . | pigz --best > ../ollama-linux-$GOARCH.tgz
RUN cd dist/linux-$GOARCH-rocm && \
    tar -cf - . | pigz --best > ../ollama-linux-$GOARCH-rocm.tgz

FROM --platform=linux/arm64 cpu-build-arm64 AS build-arm64
ENV CGO_ENABLED=1
ARG GOLANG_VERSION
WORKDIR /go/src/github.com/ollama/ollama
COPY . .
COPY --from=cuda-11-build-runner-arm64 /go/src/github.com/ollama/ollama/dist/ dist/
COPY --from=cuda-11-build-runner-arm64 /go/src/github.com/ollama/ollama/build/ build/
COPY --from=cuda-12-build-runner-arm64 /go/src/github.com/ollama/ollama/dist/ dist/
COPY --from=cuda-12-build-runner-arm64 /go/src/github.com/ollama/ollama/build/ build/
ARG GOFLAGS
ARG CGO_CFLAGS
RUN --mount=type=cache,target=/root/.ccache \
    go build -trimpath -o dist/linux-arm64/bin/ollama .
RUN cd dist/linux-$GOARCH && \
    tar --exclude runners -cf - . | pigz --best > ../ollama-linux-$GOARCH.tgz

FROM --platform=linux/amd64 scratch AS dist-amd64
COPY --from=build-amd64 /go/src/github.com/ollama/ollama/dist/ollama-linux-*.tgz /
FROM --platform=linux/arm64 scratch AS dist-arm64
COPY --from=build-arm64 /go/src/github.com/ollama/ollama/dist/ollama-linux-*.tgz /
FROM dist-$TARGETARCH as dist


# Optimized container images do not cary nested payloads
FROM --platform=linux/amd64 cpu-builder-amd64 AS container-build-amd64
WORKDIR /go/src/github.com/ollama/ollama
COPY . .
ARG GOFLAGS
ARG CGO_CFLAGS
RUN --mount=type=cache,target=/root/.ccache \
    go build -trimpath -o dist/linux-amd64/bin/ollama .

FROM --platform=linux/arm64 cpu-builder-arm64 AS container-build-arm64
WORKDIR /go/src/github.com/ollama/ollama
COPY . .
ARG GOFLAGS
ARG CGO_CFLAGS
RUN --mount=type=cache,target=/root/.ccache \
    go build -trimpath -o dist/linux-arm64/bin/ollama .

FROM --platform=linux/amd64 ubuntu:22.04 AS runtime-amd64
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=container-build-amd64 /go/src/github.com/ollama/ollama/dist/linux-amd64/bin/ /bin/
COPY --from=cpu-build-amd64 /go/src/github.com/ollama/ollama/dist/linux-amd64/lib/ /lib/
COPY --from=cpu_avx-build-amd64 /go/src/github.com/ollama/ollama/dist/linux-amd64/lib/ /lib/
COPY --from=cpu_avx2-build-amd64 /go/src/github.com/ollama/ollama/dist/linux-amd64/lib/ /lib/
COPY --from=cuda-11-build-amd64 /go/src/github.com/ollama/ollama/dist/linux-amd64/lib/ /lib/
COPY --from=cuda-12-build-amd64 /go/src/github.com/ollama/ollama/dist/linux-amd64/lib/ /lib/

FROM --platform=linux/arm64 ubuntu:22.04 AS runtime-arm64
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=container-build-arm64 /go/src/github.com/ollama/ollama/dist/linux-arm64/bin/ /bin/
COPY --from=cpu-build-arm64 /go/src/github.com/ollama/ollama/dist/linux-arm64/lib/ /lib/
COPY --from=cuda-11-build-runner-arm64 /go/src/github.com/ollama/ollama/dist/linux-arm64/lib/ /lib/
COPY --from=cuda-12-build-runner-arm64 /go/src/github.com/ollama/ollama/dist/linux-arm64/lib/ /lib/

# ROCm libraries larger so we keep it distinct from the CPU/CUDA image
FROM --platform=linux/amd64 ubuntu:22.04 AS runtime-rocm
# Frontload the rocm libraries which are large, and rarely change to increase chance of a common layer
# across releases
COPY --from=rocm-build-amd64 /go/src/github.com/ollama/ollama/dist/linux-amd64-rocm/lib/ /lib/
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=container-build-amd64 /go/src/github.com/ollama/ollama/dist/linux-amd64/bin/ /bin/
COPY --from=cpu-build-amd64 /go/src/github.com/ollama/ollama/dist/linux-amd64/lib/ /lib/
COPY --from=cpu_avx-build-amd64 /go/src/github.com/ollama/ollama/dist/linux-amd64/lib/ /lib/
COPY --from=cpu_avx2-build-amd64 /go/src/github.com/ollama/ollama/dist/linux-amd64/lib/ /lib/
COPY --from=rocm-build-amd64 /go/src/github.com/ollama/ollama/dist/linux-amd64/lib/ /lib/
EXPOSE 11434
ENV OLLAMA_HOST=0.0.0.0

ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]

FROM runtime-$TARGETARCH
EXPOSE 11434
ENV OLLAMA_HOST=0.0.0.0
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all

ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]
