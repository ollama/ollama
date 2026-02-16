# vim: filetype=dockerfile

ARG FLAVOR=${TARGETARCH}
ARG PARALLEL=8

ARG ROCMVERSION=6.3.3
ARG JETPACK5VERSION=r35.4.1
ARG JETPACK6VERSION=r36.4.0
ARG CMAKEVERSION=3.31.2
ARG VULKANVERSION=1.4.321.1

# We require gcc v10 minimum.  v10.3 has regressions, so the rockylinux 8.5 AppStream has the latest compatible version
FROM --platform=linux/amd64 rocm/dev-almalinux-8:${ROCMVERSION}-complete AS base-amd64
RUN yum install -y yum-utils \
    && yum-config-manager --add-repo https://dl.rockylinux.org/vault/rocky/8.5/AppStream/\$basearch/os/ \
    && rpm --import https://dl.rockylinux.org/pub/rocky/RPM-GPG-KEY-Rocky-8 \
    && dnf install -y yum-utils ccache gcc-toolset-10-gcc-10.2.1-8.2.el8 gcc-toolset-10-gcc-c++-10.2.1-8.2.el8 gcc-toolset-10-binutils-2.35-11.el8 \
    && dnf install -y ccache \
    && yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
ENV PATH=/opt/rh/gcc-toolset-10/root/usr/bin:$PATH
ARG VULKANVERSION
RUN wget https://sdk.lunarg.com/sdk/download/${VULKANVERSION}/linux/vulkansdk-linux-x86_64-${VULKANVERSION}.tar.xz -O /tmp/vulkansdk-linux-x86_64-${VULKANVERSION}.tar.xz \
    && tar xvf /tmp/vulkansdk-linux-x86_64-${VULKANVERSION}.tar.xz \
    && dnf -y install ninja-build \
    && ln -s /usr/bin/python3 /usr/bin/python \  
    && /${VULKANVERSION}/vulkansdk -j 8 vulkan-headers \
    && /${VULKANVERSION}/vulkansdk -j 8 shaderc
RUN cp -r /${VULKANVERSION}/x86_64/include/* /usr/local/include/ \
    && cp -r /${VULKANVERSION}/x86_64/lib/* /usr/local/lib
ENV PATH=/${VULKANVERSION}/x86_64/bin:$PATH

FROM --platform=linux/arm64 almalinux:8 AS base-arm64
# install epel-release for ccache
RUN yum install -y yum-utils epel-release \
    && dnf install -y clang ccache git \
    && yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/sbsa/cuda-rhel8.repo
ENV CC=clang CXX=clang++

FROM base-${TARGETARCH} AS base
ARG CMAKEVERSION
RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CMAKEVERSION}/cmake-${CMAKEVERSION}-linux-$(uname -m).tar.gz | tar xz -C /usr/local --strip-components 1
ENV LDFLAGS=-s

FROM base AS cpu
RUN dnf install -y gcc-toolset-11-gcc gcc-toolset-11-gcc-c++
ENV PATH=/opt/rh/gcc-toolset-11/root/usr/bin:$PATH
ARG PARALLEL
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'CPU' \
        && cmake --build --parallel ${PARALLEL} --preset 'CPU' \
        && cmake --install build --component CPU --strip --parallel ${PARALLEL}

FROM base AS cuda-11
ARG CUDA11VERSION=11.8
RUN dnf install -y cuda-toolkit-${CUDA11VERSION//./-}
ENV PATH=/usr/local/cuda-11/bin:$PATH
ARG PARALLEL
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'CUDA 11' \
        && cmake --build --parallel ${PARALLEL} --preset 'CUDA 11' \
        && cmake --install build --component CUDA --strip --parallel ${PARALLEL}

FROM base AS cuda-12
ARG CUDA12VERSION=12.8
RUN dnf install -y cuda-toolkit-${CUDA12VERSION//./-}
ENV PATH=/usr/local/cuda-12/bin:$PATH
ARG PARALLEL
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'CUDA 12' \
        && cmake --build --parallel ${PARALLEL} --preset 'CUDA 12' \
        && cmake --install build --component CUDA --strip --parallel ${PARALLEL}


FROM base AS cuda-13
ARG CUDA13VERSION=13.0
RUN dnf install -y cuda-toolkit-${CUDA13VERSION//./-}
ENV PATH=/usr/local/cuda-13/bin:$PATH
ARG PARALLEL
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'CUDA 13' \
        && cmake --build --parallel ${PARALLEL} --preset 'CUDA 13' \
        && cmake --install build --component CUDA --strip --parallel ${PARALLEL}


FROM base AS rocm-6
ENV PATH=/opt/rocm/hcc/bin:/opt/rocm/hip/bin:/opt/rocm/bin:/opt/rocm/hcc/bin:$PATH
ARG PARALLEL
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'ROCm 6' \
        && cmake --build --parallel ${PARALLEL} --preset 'ROCm 6' \
        && cmake --install build --component HIP --strip --parallel ${PARALLEL}
RUN rm -f dist/lib/ollama/rocm/rocblas/library/*gfx90[06]*

FROM --platform=linux/arm64 nvcr.io/nvidia/l4t-jetpack:${JETPACK5VERSION} AS jetpack-5
ARG CMAKEVERSION
RUN apt-get update && apt-get install -y curl ccache \
    && curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CMAKEVERSION}/cmake-${CMAKEVERSION}-linux-$(uname -m).tar.gz | tar xz -C /usr/local --strip-components 1
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
ARG PARALLEL
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'JetPack 5' \
        && cmake --build --parallel ${PARALLEL} --preset 'JetPack 5' \
        && cmake --install build --component CUDA --strip --parallel ${PARALLEL}

FROM --platform=linux/arm64 nvcr.io/nvidia/l4t-jetpack:${JETPACK6VERSION} AS jetpack-6
ARG CMAKEVERSION
RUN apt-get update && apt-get install -y curl ccache \
    && curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CMAKEVERSION}/cmake-${CMAKEVERSION}-linux-$(uname -m).tar.gz | tar xz -C /usr/local --strip-components 1
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
ARG PARALLEL
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'JetPack 6' \
        && cmake --build --parallel ${PARALLEL} --preset 'JetPack 6' \
        && cmake --install build --component CUDA --strip --parallel ${PARALLEL}

FROM base AS vulkan
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'Vulkan' \
        && cmake --build --parallel --preset 'Vulkan' \
        && cmake --install build --component Vulkan --strip --parallel 8

FROM base AS mlx
ARG CUDA13VERSION=13.0
RUN dnf install -y cuda-toolkit-${CUDA13VERSION//./-} \
    && dnf install -y openblas-devel lapack-devel \
    && dnf install -y libcudnn9-cuda-13 libcudnn9-devel-cuda-13 \
    && dnf install -y libnccl libnccl-devel
ENV PATH=/usr/local/cuda-13/bin:$PATH
ENV BLAS_INCLUDE_DIRS=/usr/include/openblas
ENV LAPACK_INCLUDE_DIRS=/usr/include/openblas
ENV CGO_LDFLAGS="-L/usr/local/cuda-13/lib64 -L/usr/local/cuda-13/targets/x86_64-linux/lib/stubs"
ARG PARALLEL
WORKDIR /go/src/github.com/ollama/ollama
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
COPY x/ml/backend/mlx x/ml/backend/mlx
COPY go.mod go.sum .
COPY MLX_VERSION .
RUN curl -fsSL https://golang.org/dl/go$(awk '/^go/ { print $2 }' go.mod).linux-$(case $(uname -m) in x86_64) echo amd64 ;; aarch64) echo arm64 ;; esac).tar.gz | tar xz -C /usr/local
ENV PATH=/usr/local/go/bin:$PATH
RUN go mod download
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'MLX CUDA 13' -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas \
        && cmake --build --parallel ${PARALLEL} --preset 'MLX CUDA 13' \
        && cmake --install build --component MLX --strip --parallel ${PARALLEL}

FROM base AS build
WORKDIR /go/src/github.com/ollama/ollama
COPY go.mod go.sum .
RUN curl -fsSL https://golang.org/dl/go$(awk '/^go/ { print $2 }' go.mod).linux-$(case $(uname -m) in x86_64) echo amd64 ;; aarch64) echo arm64 ;; esac).tar.gz | tar xz -C /usr/local
ENV PATH=/usr/local/go/bin:$PATH
RUN go mod download
COPY . .
# Clone mlx-c headers for CGO (version from MLX_VERSION file)
RUN git clone --depth 1 --branch "$(cat MLX_VERSION)" https://github.com/ml-explore/mlx-c.git build/_deps/mlx-c-src
ARG GOFLAGS="'-ldflags=-w -s'"
ENV CGO_ENABLED=1
ENV CGO_CFLAGS="-I/go/src/github.com/ollama/ollama/build/_deps/mlx-c-src"
ARG CGO_CXXFLAGS
RUN --mount=type=cache,target=/root/.cache/go-build \
    go build -tags mlx -trimpath -buildmode=pie -o /bin/ollama .

FROM --platform=linux/amd64 scratch AS amd64
# COPY --from=cuda-11 dist/lib/ollama/ /lib/ollama/
COPY --from=cuda-12 dist/lib/ollama /lib/ollama/
COPY --from=cuda-13 dist/lib/ollama /lib/ollama/
COPY --from=vulkan  dist/lib/ollama  /lib/ollama/
COPY --from=mlx     /go/src/github.com/ollama/ollama/dist/lib/ollama /lib/ollama/

FROM --platform=linux/arm64 scratch AS arm64
# COPY --from=cuda-11 dist/lib/ollama/ /lib/ollama/
COPY --from=cuda-12 dist/lib/ollama /lib/ollama/
COPY --from=cuda-13 dist/lib/ollama/ /lib/ollama/
COPY --from=jetpack-5 dist/lib/ollama/ /lib/ollama/
COPY --from=jetpack-6 dist/lib/ollama/ /lib/ollama/

FROM scratch AS rocm
COPY --from=rocm-6 dist/lib/ollama /lib/ollama

FROM ${FLAVOR} AS archive
ARG VULKANVERSION
COPY --from=cpu dist/lib/ollama /lib/ollama
COPY --from=build /bin/ollama /bin/ollama

FROM ubuntu:24.04
RUN apt-get update \
    && apt-get install -y ca-certificates libvulkan1 libopenblas0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
COPY --from=archive /bin /usr/bin
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
COPY --from=archive /lib/ollama /usr/lib/ollama
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all
ENV OLLAMA_HOST=0.0.0.0:11434
EXPOSE 11434
ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]
