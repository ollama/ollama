# vim: filetype=dockerfile

ARG FLAVOR=${TARGETARCH}

ARG ROCMVERSION=6.1.2
ARG JETPACK5VERSION=r35.4.1
ARG JETPACK6VERSION=r36.2.0
ARG CMAKEVERSION=3.31.2

FROM --platform=linux/amd64 rocm/dev-centos-7:${ROCMVERSION}-complete AS base-amd64
RUN sed -i -e 's/mirror.centos.org/vault.centos.org/g' -e 's/^#.*baseurl=http/baseurl=http/g' -e 's/^mirrorlist=http/#mirrorlist=http/g' /etc/yum.repos.d/*.repo \
    && yum install -y yum-utils devtoolset-10-gcc devtoolset-10-gcc-c++ \
    && yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo \
    && curl -s -L https://github.com/ccache/ccache/releases/download/v4.10.2/ccache-4.10.2-linux-x86_64.tar.xz | tar -Jx -C /usr/local/bin --strip-components 1
ENV PATH=/opt/rh/devtoolset-10/root/usr/bin:/opt/rh/devtoolset-11/root/usr/bin:$PATH

FROM --platform=linux/arm64 rockylinux:8 AS base-arm64
# install epel-release for ccache
RUN yum install -y yum-utils epel-release \
    && yum install -y clang ccache \
    && yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/sbsa/cuda-rhel8.repo
ENV CC=clang CXX=clang++

FROM base-${TARGETARCH} AS base
ARG CMAKEVERSION
RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CMAKEVERSION}/cmake-${CMAKEVERSION}-linux-$(uname -m).tar.gz | tar xz -C /usr/local --strip-components 1
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
ENV LDFLAGS=-s

FROM base AS cpu
# amd64 uses gcc which requires devtoolset-11 for AVX extensions while arm64 uses clang
RUN if [ "$(uname -m)" = "x86_64" ]; then yum install -y devtoolset-11-gcc devtoolset-11-gcc-c++; fi
ENV PATH=/opt/rh/devtoolset-11/root/usr/bin:$PATH
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'Default' && cmake --build --parallel --preset 'Default'

FROM base AS cuda-11
ARG CUDA11VERSION=11.3
RUN yum install -y cuda-toolkit-${CUDA11VERSION//./-}
ENV PATH=/usr/local/cuda-11/bin:$PATH
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'CUDA 11' && cmake --build --parallel --preset 'CUDA 11'

FROM base AS cuda-12
ARG CUDA12VERSION=12.4
RUN yum install -y cuda-toolkit-${CUDA12VERSION//./-}
ENV PATH=/usr/local/cuda-12/bin:$PATH
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'CUDA 12' && cmake --build --parallel --preset 'CUDA 12'

FROM base AS rocm-6
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'ROCm 6' && cmake --build --parallel --preset 'ROCm 6'

FROM --platform=linux/arm64 nvcr.io/nvidia/l4t-jetpack:${JETPACK5VERSION} AS jetpack-5
ARG CMAKEVERSION
RUN apt-get update && apt-get install -y curl ccache \
    && curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CMAKEVERSION}/cmake-${CMAKEVERSION}-linux-$(uname -m).tar.gz | tar xz -C /usr/local --strip-components 1
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'JetPack 5' && cmake --build --parallel --preset 'JetPack 5'

FROM --platform=linux/arm64 nvcr.io/nvidia/l4t-jetpack:${JETPACK6VERSION} AS jetpack-6
ARG CMAKEVERSION
RUN apt-get update && apt-get install -y curl ccache \
    && curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CMAKEVERSION}/cmake-${CMAKEVERSION}-linux-$(uname -m).tar.gz | tar xz -C /usr/local --strip-components 1
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'JetPack 6' && cmake --build --parallel --preset 'JetPack 6'

FROM base AS build
ARG GOVERSION=1.23.4
RUN curl -fsSL https://golang.org/dl/go${GOVERSION}.linux-$(case $(uname -m) in x86_64) echo amd64 ;; aarch64) echo arm64 ;; esac).tar.gz | tar xz -C /usr/local
ENV PATH=/usr/local/go/bin:$PATH
WORKDIR /go/src/github.com/ollama/ollama
COPY . .
ARG GOFLAGS="'-ldflags=-w -s'"
ENV CGO_ENABLED=1
RUN --mount=type=cache,target=/root/.cache/go-build \
    go build -trimpath -buildmode=pie -o /bin/ollama .

FROM --platform=linux/amd64 scratch AS amd64
COPY --from=cuda-11 --chmod=644 \
    dist/build/lib/libggml-cuda.so \
    /usr/local/cuda/lib64/libcublas.so.11 \
    /usr/local/cuda/lib64/libcublasLt.so.11 \
    /usr/local/cuda/lib64/libcudart.so.11.0 \
    /lib/ollama/cuda_v11/
COPY --from=cuda-12 --chmod=644 \
    dist/build/lib/libggml-cuda.so \
    /usr/local/cuda/lib64/libcublas.so.12 \
    /usr/local/cuda/lib64/libcublasLt.so.12 \
    /usr/local/cuda/lib64/libcudart.so.12 \
    /lib/ollama/cuda_v12/

FROM --platform=linux/arm64 scratch AS arm64
COPY --from=cuda-11 --chmod=644 \
    dist/build/lib/libggml-cuda.so \
    /usr/local/cuda/lib64/libcublas.so.11 \
    /usr/local/cuda/lib64/libcublasLt.so.11 \
    /usr/local/cuda/lib64/libcudart.so.11.0 \
    /lib/ollama/cuda_v11/
COPY --from=cuda-12 --chmod=644 \
    dist/build/lib/libggml-cuda.so \
    /usr/local/cuda/lib64/libcublas.so.12 \
    /usr/local/cuda/lib64/libcublasLt.so.12 \
    /usr/local/cuda/lib64/libcudart.so.12 \
    /lib/ollama/cuda_v12/
COPY --from=jetpack-5 --chmod=644 \
    dist/build/lib/libggml-cuda.so \
    /usr/local/cuda/lib64/libcublas.so.11 \
    /usr/local/cuda/lib64/libcublasLt.so.11 \
    /usr/local/cuda/lib64/libcudart.so.11.0 \
    /lib/ollama/cuda_jetpack5/
COPY --from=jetpack-6 --chmod=644 \
    dist/build/lib/libggml-cuda.so \
    /usr/local/cuda/lib64/libcublas.so.12 \
    /usr/local/cuda/lib64/libcublasLt.so.12 \
    /usr/local/cuda/lib64/libcudart.so.12 \
    /lib/ollama/cuda_jetpack6/

FROM --platform=linux/arm64 scratch AS rocm
COPY --from=rocm-6 --chmod=644 \
    dist/build/lib/libggml-hip.so \
    /opt/rocm/lib/libamdhip64.so.6 \
    /opt/rocm/lib/libhipblas.so.2 \
    /opt/rocm/lib/librocblas.so.4 \
    /opt/rocm/lib/libamd_comgr.so.2 \
    /opt/rocm/lib/libhsa-runtime64.so.1 \
    /opt/rocm/lib/librocprofiler-register.so.0 \
    /opt/amdgpu/lib64/libdrm_amdgpu.so.1 \
    /opt/amdgpu/lib64/libdrm.so.2 \
    /usr/lib64/libnuma.so.1 \
    /lib/ollama/rocm/
COPY --from=rocm-6 /opt/rocm/lib/rocblas/ /lib/ollama/rocm/rocblas/

FROM ${FLAVOR} AS archive
COPY --from=cpu --chmod=644 \
    dist/build/lib/libggml-base.so \
    dist/build/lib/libggml-cpu-*.so \
    /lib/ollama/
COPY --from=build /bin/ollama /bin/ollama

FROM ubuntu:20.04
RUN apt-get update \
    && apt-get install -y ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
COPY --from=archive /bin/ /usr/bin/
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
COPY --from=archive /lib/ollama/ /usr/lib/ollama/
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/ollama
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all
ENV OLLAMA_HOST=0.0.0.0:11434
EXPOSE 11434
ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]
