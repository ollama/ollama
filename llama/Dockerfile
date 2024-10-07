# Note: once we have fully transitioned to the Go server, this will replace the old Dockerfile at the top of the tree
ARG GOLANG_VERSION=1.22.5
ARG CMAKE_VERSION=3.22.1
ARG CUDA_VERSION_11=11.3.1
ARG CUDA_V11_ARCHITECTURES="50;52;53;60;61;62;70;72;75;80;86"
ARG CUDA_VERSION_12=12.4.0
ARG CUDA_V12_ARCHITECTURES="60;61;62;70;72;75;80;86;87;89;90;90a"
ARG ROCM_VERSION=6.1.2

### To create a local image for building linux binaries on mac or windows with efficient incremental builds
#
# docker build --platform linux/amd64 -t builder-amd64 -f Dockerfile.new --target unified-builder-amd64 .
# docker run --platform linux/amd64 --rm -it -v $(pwd):/go/src/github.com/ollama/ollama/ builder-amd64
#
### Then incremental builds will be much faster in this container
#
# make -C llama -j 10 && go build -trimpath -o dist/linux-amd64/ollama .
#
FROM --platform=linux/amd64 rocm/dev-centos-7:${ROCM_VERSION}-complete AS unified-builder-amd64
ARG CMAKE_VERSION
ARG GOLANG_VERSION
ARG CUDA_VERSION_11
ARG CUDA_VERSION_12
COPY ./scripts/rh_linux_deps.sh /
ENV PATH /opt/rh/devtoolset-10/root/usr/bin:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/opt/amdgpu/lib64
RUN CMAKE_VERSION=${CMAKE_VERSION} GOLANG_VERSION=${GOLANG_VERSION} sh /rh_linux_deps.sh
RUN yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo && \
    dnf clean all && \
    dnf install -y \
    zsh \
    cuda-$(echo ${CUDA_VERSION_11} | cut -f1-2 -d. | sed -e "s/\./-/g") \
    cuda-$(echo ${CUDA_VERSION_12} | cut -f1-2 -d. | sed -e "s/\./-/g")
# TODO intel oneapi goes here...
ENV GOARCH amd64
ENV CGO_ENABLED 1
WORKDIR /go/src/github.com/ollama/ollama/
ENTRYPOINT [ "zsh" ]

### To create a local image for building linux binaries on mac or linux/arm64 with efficient incremental builds
# Note: this does not contain jetson variants
#
# docker build --platform linux/arm64 -t builder-arm64 -f Dockerfile.new --target unified-builder-arm64 .
# docker run --platform linux/arm64 --rm -it -v $(pwd):/go/src/github.com/ollama/ollama/ builder-arm64
#
FROM --platform=linux/arm64 rockylinux:8 AS unified-builder-arm64
ARG CMAKE_VERSION
ARG GOLANG_VERSION
ARG CUDA_VERSION_11
ARG CUDA_VERSION_12
COPY ./scripts/rh_linux_deps.sh /
RUN CMAKE_VERSION=${CMAKE_VERSION} GOLANG_VERSION=${GOLANG_VERSION} sh /rh_linux_deps.sh
RUN yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/sbsa/cuda-rhel8.repo && \
    dnf config-manager --set-enabled appstream && \
    dnf clean all && \
    dnf install -y \
    zsh \
    cuda-toolkit-$(echo ${CUDA_VERSION_11} | cut -f1-2 -d. | sed -e "s/\./-/g") \
    cuda-toolkit-$(echo ${CUDA_VERSION_12} | cut -f1-2 -d. | sed -e "s/\./-/g")
ENV PATH /opt/rh/gcc-toolset-10/root/usr/bin:$PATH:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/opt/amdgpu/lib64
ENV GOARCH amd64
ENV CGO_ENABLED 1
WORKDIR /go/src/github.com/ollama/ollama/
ENTRYPOINT [ "zsh" ]

FROM --platform=linux/amd64 unified-builder-amd64 AS runners-amd64
COPY . .
ARG OLLAMA_SKIP_CUDA_GENERATE
ARG OLLAMA_SKIP_CUDA_11_GENERATE
ARG OLLAMA_SKIP_CUDA_12_GENERATE
ARG OLLAMA_SKIP_ROCM_GENERATE
ARG CUDA_V11_ARCHITECTURES
ARG CUDA_V12_ARCHITECTURES
ARG OLLAMA_FAST_BUILD
RUN --mount=type=cache,target=/root/.ccache \
    if grep "^flags" /proc/cpuinfo|grep avx>/dev/null; then \
        make -C llama -j $(expr $(nproc) / 2 ) ; \
    else \
        make -C llama -j 5 ; \
    fi

FROM --platform=linux/arm64 unified-builder-arm64 AS runners-arm64
COPY . .
ARG OLLAMA_SKIP_CUDA_GENERATE
ARG OLLAMA_SKIP_CUDA_11_GENERATE
ARG OLLAMA_SKIP_CUDA_12_GENERATE
ARG CUDA_V11_ARCHITECTURES
ARG CUDA_V12_ARCHITECTURES
ARG OLLAMA_FAST_BUILD
RUN --mount=type=cache,target=/root/.ccache \
    make -C llama -j 8


# Intermediate stages used for ./scripts/build_linux.sh
FROM --platform=linux/amd64 centos:7 AS builder-amd64
ARG CMAKE_VERSION
ARG GOLANG_VERSION
COPY ./scripts/rh_linux_deps.sh /
RUN CMAKE_VERSION=${CMAKE_VERSION} GOLANG_VERSION=${GOLANG_VERSION} sh /rh_linux_deps.sh
ENV PATH /opt/rh/devtoolset-10/root/usr/bin:$PATH
ENV CGO_ENABLED 1
ENV GOARCH amd64
WORKDIR /go/src/github.com/ollama/ollama

FROM --platform=linux/amd64 builder-amd64 AS build-amd64
COPY . .
COPY --from=runners-amd64 /go/src/github.com/ollama/ollama/dist/ dist/
COPY --from=runners-amd64 /go/src/github.com/ollama/ollama/build/ build/
ARG GOFLAGS
ARG CGO_CFLAGS
ARG OLLAMA_SKIP_ROCM_GENERATE
RUN --mount=type=cache,target=/root/.ccache \
    go build -trimpath -o dist/linux-amd64/bin/ollama .
RUN cd dist/linux-$GOARCH && \
    tar --exclude runners -cf - . | pigz --best > ../ollama-linux-$GOARCH.tgz
RUN if [ -z ${OLLAMA_SKIP_ROCM_GENERATE} ] ; then \
    cd dist/linux-$GOARCH-rocm && \
    tar -cf - . | pigz --best > ../ollama-linux-$GOARCH-rocm.tgz ;\
    fi

FROM --platform=linux/arm64 rockylinux:8 AS builder-arm64
ARG CMAKE_VERSION
ARG GOLANG_VERSION
COPY ./scripts/rh_linux_deps.sh /
RUN CMAKE_VERSION=${CMAKE_VERSION} GOLANG_VERSION=${GOLANG_VERSION} sh /rh_linux_deps.sh
ENV PATH /opt/rh/gcc-toolset-10/root/usr/bin:$PATH
ENV CGO_ENABLED 1
ENV GOARCH arm64
WORKDIR /go/src/github.com/ollama/ollama

FROM --platform=linux/arm64 builder-arm64 AS build-arm64
COPY . .
COPY --from=runners-arm64 /go/src/github.com/ollama/ollama/dist/ dist/
COPY --from=runners-arm64 /go/src/github.com/ollama/ollama/build/ build/
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
FROM dist-$TARGETARCH AS dist


# Optimized container images do not cary nested payloads
FROM --platform=linux/amd64 builder-amd64 AS container-build-amd64
WORKDIR /go/src/github.com/ollama/ollama
COPY . .
ARG GOFLAGS
ARG CGO_CFLAGS
RUN --mount=type=cache,target=/root/.ccache \
    go build -trimpath -o dist/linux-amd64/bin/ollama .

FROM --platform=linux/arm64 builder-arm64 AS container-build-arm64
WORKDIR /go/src/github.com/ollama/ollama
COPY . .
ARG GOFLAGS
ARG CGO_CFLAGS
RUN --mount=type=cache,target=/root/.ccache \
    go build -trimpath -o dist/linux-arm64/bin/ollama .

# For amd64 container images, filter out cuda/rocm to minimize size
FROM runners-amd64 AS runners-cuda-amd64
RUN rm -rf \
    ./dist/linux-amd64/lib/ollama/libggml_hipblas.so \
    ./dist/linux-amd64/lib/ollama/runners/rocm*

FROM runners-amd64 AS runners-rocm-amd64
RUN rm -rf \
    ./dist/linux-amd64/lib/ollama/libggml_cuda*.so \
    ./dist/linux-amd64/lib/ollama/libcu*.so* \
    ./dist/linux-amd64/lib/ollama/runners/cuda*

FROM --platform=linux/amd64 ubuntu:22.04 AS runtime-amd64
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    rm -rf /var/lib/apt/lists/*
COPY --from=container-build-amd64 /go/src/github.com/ollama/ollama/dist/linux-amd64/bin/ /bin/
COPY --from=runners-cuda-amd64 /go/src/github.com/ollama/ollama/dist/linux-amd64/lib/ /lib/

FROM --platform=linux/arm64 ubuntu:22.04 AS runtime-arm64
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    rm -rf /var/lib/apt/lists/*
COPY --from=container-build-arm64 /go/src/github.com/ollama/ollama/dist/linux-arm64/bin/ /bin/
COPY --from=runners-arm64 /go/src/github.com/ollama/ollama/dist/linux-arm64/lib/ /lib/

# ROCm libraries larger so we keep it distinct from the CPU/CUDA image
FROM --platform=linux/amd64 ubuntu:22.04 AS runtime-rocm
# Frontload the rocm libraries which are large, and rarely change to increase chance of a common layer
# across releases
COPY --from=build-amd64 /go/src/github.com/ollama/ollama/dist/linux-amd64-rocm/lib/ /lib/
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    rm -rf /var/lib/apt/lists/*
COPY --from=container-build-amd64 /go/src/github.com/ollama/ollama/dist/linux-amd64/bin/ /bin/
COPY --from=runners-rocm-amd64 /go/src/github.com/ollama/ollama/dist/linux-amd64/lib/ /lib/

EXPOSE 11434
ENV OLLAMA_HOST 0.0.0.0

ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]

FROM runtime-$TARGETARCH
EXPOSE 11434
ENV OLLAMA_HOST 0.0.0.0
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all

ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]
