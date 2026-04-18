# vim: filetype=dockerfile

ARG FLAVOR=${TARGETARCH}

ARG ROCMVERSION=7.2.1
ARG JETPACK5VERSION=r35.4.1
ARG JETPACK6VERSION=r36.4.0
ARG CMAKEVERSION=3.31.2
ARG NINJAVERSION=1.12.1
ARG VULKANVERSION=1.4.321.1

# Default empty stages for local MLX source overrides.
# Override with: docker build --build-context local-mlx=../mlx --build-context local-mlx-c=../mlx-c
FROM scratch AS local-mlx
FROM scratch AS local-mlx-c

FROM --platform=linux/amd64 rocm/dev-almalinux-8:${ROCMVERSION}-complete AS base-amd64
RUN dnf install -y yum-utils ccache gcc-toolset-11-gcc gcc-toolset-11-gcc-c++ gcc-toolset-11-binutils \
    && yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
ENV PATH=/opt/rh/gcc-toolset-11/root/usr/bin:$PATH

FROM --platform=linux/arm64 almalinux:8 AS base-arm64
# install epel-release for ccache
RUN yum install -y yum-utils epel-release \
    && dnf install -y clang ccache git \
    && yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/sbsa/cuda-rhel8.repo
ENV CC=clang CXX=clang++

FROM base-${TARGETARCH} AS base
ARG CMAKEVERSION
ARG NINJAVERSION
RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CMAKEVERSION}/cmake-${CMAKEVERSION}-linux-$(uname -m).tar.gz | tar xz -C /usr/local --strip-components 1
RUN dnf install -y unzip \
    && curl -fsSL -o /tmp/ninja.zip https://github.com/ninja-build/ninja/releases/download/v${NINJAVERSION}/ninja-linux$([ "$(uname -m)" = "aarch64" ] && echo "-aarch64").zip \
    && unzip /tmp/ninja.zip -d /usr/local/bin \
    && rm /tmp/ninja.zip
ENV CMAKE_GENERATOR=Ninja
ENV LDFLAGS=-s

#
# GPU toolchain stages — provide compilers for llama-server GPU builds
#

FROM base AS cpu-deps
RUN dnf install -y gcc-toolset-11-gcc gcc-toolset-11-gcc-c++
ENV PATH=/opt/rh/gcc-toolset-11/root/usr/bin:$PATH

FROM base AS cuda-12-deps
ARG CUDA12VERSION=12.8
RUN dnf install -y cuda-toolkit-${CUDA12VERSION//./-}
ENV PATH=/usr/local/cuda-12/bin:$PATH

FROM base AS cuda-13-deps
ARG CUDA13VERSION=13.0
RUN dnf install -y cuda-toolkit-${CUDA13VERSION//./-}
ENV PATH=/usr/local/cuda-13/bin:$PATH

FROM base AS rocm-7-deps
ENV PATH=/opt/rocm/llvm/bin:/opt/rocm/hcc/bin:/opt/rocm/hip/bin:/opt/rocm/bin:$PATH

FROM base AS vulkan-deps
ARG VULKANVERSION
RUN ln -s /usr/bin/python3 /usr/bin/python \
    && wget https://sdk.lunarg.com/sdk/download/${VULKANVERSION}/linux/vulkansdk-linux-x86_64-${VULKANVERSION}.tar.xz -O /tmp/vulkansdk.tar.xz \
    && tar xvf /tmp/vulkansdk.tar.xz -C /tmp \
    && /tmp/${VULKANVERSION}/vulkansdk -j 8 vulkan-headers \
    && /tmp/${VULKANVERSION}/vulkansdk -j 8 shaderc \
    && cp -r /tmp/${VULKANVERSION}/x86_64/include/* /usr/local/include/ \
    && cp -r /tmp/${VULKANVERSION}/x86_64/lib/* /usr/local/lib \
    && cp -r /tmp/${VULKANVERSION}/x86_64/bin/* /usr/local/bin/ \
    && rm -rf /tmp/${VULKANVERSION} /tmp/vulkansdk.tar.xz

#
# llama-server stages — rebuild when LLAMA_CPP_VERSION or llama/server/ changes.
#
# CPU stage: llama-server + ggml-base + ggml-cpu variants → lib/ollama/
# GPU stages: GPU backend .so only → lib/ollama/<variant>/
#

FROM cpu-deps AS llama-server-cpu
COPY LLAMA_CPP_VERSION .
COPY llama/server llama/server
RUN --mount=type=cache,target=/root/.ccache \
    cmake -S llama/server --preset cpu \
        && cmake --build build/llama-server-cpu -- -l $(nproc) \
        && cmake --install build/llama-server-cpu --component llama-server --strip

FROM cuda-12-deps AS llama-server-cuda-v12
COPY LLAMA_CPP_VERSION .
COPY llama/server llama/server
RUN --mount=type=cache,target=/root/.ccache \
    cmake -S llama/server --preset cuda-v12 \
        && cmake --build build/llama-server-cuda-v12 -- -l $(nproc) \
        && cmake --install build/llama-server-cuda-v12 --component llama-server --strip

FROM cuda-13-deps AS llama-server-cuda-v13
COPY LLAMA_CPP_VERSION .
COPY llama/server llama/server
RUN --mount=type=cache,target=/root/.ccache \
    cmake -S llama/server --preset cuda-v13 \
        && cmake --build build/llama-server-cuda-v13 -- -l $(nproc) \
        && cmake --install build/llama-server-cuda-v13 --component llama-server --strip

FROM rocm-7-deps AS llama-server-rocm
ENV CC=clang CXX=clang++
COPY LLAMA_CPP_VERSION .
COPY llama/server llama/server
RUN --mount=type=cache,target=/root/.ccache \
    cmake -S llama/server --preset rocm \
        -DAMDGPU_TARGETS="gfx942;gfx950;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1200;gfx1201;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-" \
        -DCMAKE_HIP_FLAGS="-parallel-jobs=4" \
        && cmake --build build/llama-server-rocm -- -l $(nproc) \
        && cmake --install build/llama-server-rocm --component llama-server --strip
RUN rm -f dist/lib/ollama/rocm/rocblas/library/*gfx90[06]*

FROM vulkan-deps AS llama-server-vulkan
COPY LLAMA_CPP_VERSION .
COPY llama/server llama/server
RUN --mount=type=cache,target=/root/.ccache \
    cmake -S llama/server --preset vulkan \
        && cmake --build build/llama-server-vulkan -- -l $(nproc) \
        && cmake --install build/llama-server-vulkan --component llama-server --strip

#
# JetPack stages — self-contained with their own base images
#

FROM --platform=linux/arm64 nvcr.io/nvidia/l4t-jetpack:${JETPACK5VERSION} AS jetpack-5
ARG CMAKEVERSION
ARG NINJAVERSION
RUN apt-get update && apt-get install -y curl ccache git unzip \
    && curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CMAKEVERSION}/cmake-${CMAKEVERSION}-linux-$(uname -m).tar.gz | tar xz -C /usr/local --strip-components 1 \
    && curl -fsSL -o /tmp/ninja.zip https://github.com/ninja-build/ninja/releases/download/v${NINJAVERSION}/ninja-linux-aarch64.zip \
    && unzip /tmp/ninja.zip -d /usr/local/bin \
    && rm /tmp/ninja.zip
ENV CMAKE_GENERATOR=Ninja
COPY LLAMA_CPP_VERSION .
COPY llama/server llama/server
RUN --mount=type=cache,target=/root/.ccache \
    cmake -S llama/server --preset jetpack5 \
        && cmake --build build/llama-server-jetpack5 -- -l $(nproc) \
        && cmake --install build/llama-server-jetpack5 --component llama-server --strip

FROM --platform=linux/arm64 nvcr.io/nvidia/l4t-jetpack:${JETPACK6VERSION} AS jetpack-6
ARG CMAKEVERSION
ARG NINJAVERSION
RUN apt-get update && apt-get install -y curl ccache git unzip \
    && curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CMAKEVERSION}/cmake-${CMAKEVERSION}-linux-$(uname -m).tar.gz | tar xz -C /usr/local --strip-components 1 \
    && curl -fsSL -o /tmp/ninja.zip https://github.com/ninja-build/ninja/releases/download/v${NINJAVERSION}/ninja-linux-aarch64.zip \
    && unzip /tmp/ninja.zip -d /usr/local/bin \
    && rm /tmp/ninja.zip
ENV CMAKE_GENERATOR=Ninja
COPY LLAMA_CPP_VERSION .
COPY llama/server llama/server
RUN --mount=type=cache,target=/root/.ccache \
    cmake -S llama/server --preset jetpack6 \
        && cmake --build build/llama-server-jetpack6 -- -l $(nproc) \
        && cmake --install build/llama-server-jetpack6 --component llama-server --strip

#
# MLX stage
#

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
WORKDIR /go/src/github.com/ollama/ollama
COPY CMakeLists.txt CMakePresets.json .
COPY x/imagegen/mlx x/imagegen/mlx
COPY go.mod go.sum .
COPY MLX_VERSION MLX_C_VERSION .
RUN curl -fsSL https://golang.org/dl/go$(awk '/^go/ { print $2 }' go.mod).linux-$(case $(uname -m) in x86_64) echo amd64 ;; aarch64) echo arm64 ;; esac).tar.gz | tar xz -C /usr/local
ENV PATH=/usr/local/go/bin:$PATH
RUN go mod download
RUN --mount=type=cache,target=/root/.ccache \
    --mount=type=bind,from=local-mlx,target=/tmp/local-mlx \
    --mount=type=bind,from=local-mlx-c,target=/tmp/local-mlx-c \
    if [ -f /tmp/local-mlx/CMakeLists.txt ]; then \
        export OLLAMA_MLX_SOURCE=/tmp/local-mlx; \
    fi \
    && if [ -f /tmp/local-mlx-c/CMakeLists.txt ]; then \
        export OLLAMA_MLX_C_SOURCE=/tmp/local-mlx-c; \
    fi \
    && cmake --preset 'MLX CUDA 13' -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas \
        && cmake --build --preset 'MLX CUDA 13' -- -l $(nproc) \
        && cmake --install build --component MLX --strip

#
# Go build
#

FROM base AS build
WORKDIR /go/src/github.com/ollama/ollama
COPY go.mod go.sum .
RUN curl -fsSL https://golang.org/dl/go$(awk '/^go/ { print $2 }' go.mod).linux-$(case $(uname -m) in x86_64) echo amd64 ;; aarch64) echo arm64 ;; esac).tar.gz | tar xz -C /usr/local
ENV PATH=/usr/local/go/bin:$PATH
RUN go mod download
COPY . .
ARG GOFLAGS="'-ldflags=-w -s'"
ENV CGO_ENABLED=1
ARG CGO_CFLAGS
ARG CGO_CXXFLAGS
ENV CGO_CFLAGS="${CGO_CFLAGS}"
ENV CGO_CXXFLAGS="${CGO_CXXFLAGS}"
RUN --mount=type=cache,target=/root/.cache/go-build \
    go build -trimpath -buildmode=pie -o /bin/ollama .

#
# Assembly stages — combine llama-server variants + GPU runtime libs
#

FROM --platform=linux/amd64 scratch AS amd64
COPY --from=llama-server-cpu      dist/lib/ollama /lib/ollama/
COPY --from=llama-server-cuda-v12 dist/lib/ollama /lib/ollama/
COPY --from=llama-server-cuda-v13 dist/lib/ollama /lib/ollama/
COPY --from=llama-server-vulkan   dist/lib/ollama /lib/ollama/
COPY --from=mlx     /go/src/github.com/ollama/ollama/dist/lib/ollama /lib/ollama/

FROM --platform=linux/arm64 scratch AS arm64
COPY --from=llama-server-cpu dist/lib/ollama /lib/ollama/
COPY --from=jetpack-5 dist/lib/ollama/ /lib/ollama/
COPY --from=jetpack-6 dist/lib/ollama/ /lib/ollama/

FROM scratch AS rocm
COPY --from=llama-server-cpu  dist/lib/ollama /lib/ollama
COPY --from=llama-server-rocm dist/lib/ollama /lib/ollama

FROM ${FLAVOR} AS archive
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
