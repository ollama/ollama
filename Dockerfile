ARG GOLANG_VERSION=1.22.8
ARG CUDA_VERSION_11=11.3.1
ARG CUDA_VERSION_12=12.4.0
ARG ROCM_VERSION=6.2.4
ARG JETPACK_6=r36.2.0
ARG JETPACK_5=r35.4.1

### To create a local image for building linux binaries on mac or windows with efficient incremental builds
#
# docker build --platform linux/amd64 -t builder-amd64 -f Dockerfile --target unified-builder-amd64 .
# docker run --platform linux/amd64 --rm -it -v $(pwd):/go/src/github.com/ollama/ollama/ builder-amd64
#
### Then incremental builds will be much faster in this container
#
# make -j 10 dist
#
FROM --platform=linux/amd64 rocm/dev-centos-7:${ROCM_VERSION}-complete AS unified-builder-amd64
ARG GOLANG_VERSION
ARG CUDA_VERSION_11
ARG CUDA_VERSION_12
COPY ./scripts/rh_linux_deps.sh /
ENV PATH /opt/rh/devtoolset-10/root/usr/bin:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
RUN GOLANG_VERSION=${GOLANG_VERSION} sh /rh_linux_deps.sh
RUN yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo && \
    dnf clean all && \
    dnf install -y \
    zsh \
    cuda-toolkit-$(echo ${CUDA_VERSION_11} | cut -f1-2 -d. | sed -e "s/\./-/g") \
    cuda-toolkit-$(echo ${CUDA_VERSION_12} | cut -f1-2 -d. | sed -e "s/\./-/g")
# TODO intel oneapi goes here...
ENV GOARCH amd64
ENV CGO_ENABLED 1
WORKDIR /go/src/github.com/ollama/ollama/
ENTRYPOINT [ "zsh" ]

### To create a local image for building linux binaries on mac or linux/arm64 with efficient incremental builds
# Note: this does not contain jetson variants
#
# docker build --platform linux/arm64 -t builder-arm64 -f Dockerfile --target unified-builder-arm64 .
# docker run --platform linux/arm64 --rm -it -v $(pwd):/go/src/github.com/ollama/ollama/ builder-arm64
#
FROM --platform=linux/arm64 rockylinux:8 AS unified-builder-arm64
ARG GOLANG_VERSION
ARG CUDA_VERSION_11
ARG CUDA_VERSION_12
COPY ./scripts/rh_linux_deps.sh /
RUN GOLANG_VERSION=${GOLANG_VERSION} sh /rh_linux_deps.sh
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
ENV GOARCH arm64
ENV CGO_ENABLED 1
WORKDIR /go/src/github.com/ollama/ollama/
ENTRYPOINT [ "zsh" ]

FROM --platform=linux/amd64 unified-builder-amd64 AS build-amd64
COPY . .
ARG OLLAMA_SKIP_CUDA_GENERATE
ARG OLLAMA_SKIP_ROCM_GENERATE
ARG OLLAMA_FAST_BUILD
ARG VERSION
RUN --mount=type=cache,target=/root/.ccache \
    if grep "^flags" /proc/cpuinfo|grep avx>/dev/null; then \
        make -j $(nproc) dist ; \
    else \
        make -j 5 dist ; \
    fi
RUN cd dist/linux-$GOARCH && \
    tar -cf - . | pigz --best > ../ollama-linux-$GOARCH.tgz
RUN if [ -z ${OLLAMA_SKIP_ROCM_GENERATE} ] ; then \
    cd dist/linux-$GOARCH-rocm && \
    tar -cf - . | pigz --best > ../ollama-linux-$GOARCH-rocm.tgz ;\
    fi

# Jetsons need to be built in discrete stages
FROM --platform=linux/arm64 nvcr.io/nvidia/l4t-jetpack:${JETPACK_5} AS runners-jetpack5-arm64
ARG GOLANG_VERSION
RUN apt-get update && apt-get install -y git curl ccache && \
    curl -s -L https://dl.google.com/go/go${GOLANG_VERSION}.linux-arm64.tar.gz | tar xz -C /usr/local && \
    ln -s /usr/local/go/bin/go /usr/local/bin/go && \
    ln -s /usr/local/go/bin/gofmt /usr/local/bin/gofmt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /go/src/github.com/ollama/ollama/
COPY . .
ARG CGO_CFLAGS
ENV GOARCH arm64
ARG VERSION
RUN --mount=type=cache,target=/root/.ccache \
    make -j 5 dist_cuda_v11 \
        CUDA_ARCHITECTURES="72;87" \
        GPU_RUNNER_VARIANT=_jetpack5 \
        DIST_LIB_DIR=/go/src/github.com/ollama/ollama/dist/linux-arm64-jetpack5/lib/ollama \
        DIST_GPU_RUNNER_DEPS_DIR=/go/src/github.com/ollama/ollama/dist/linux-arm64-jetpack5/lib/ollama/cuda_jetpack5

FROM --platform=linux/arm64 nvcr.io/nvidia/l4t-jetpack:${JETPACK_6} AS runners-jetpack6-arm64
ARG GOLANG_VERSION
RUN apt-get update && apt-get install -y git curl ccache && \
    curl -s -L https://dl.google.com/go/go${GOLANG_VERSION}.linux-arm64.tar.gz | tar xz -C /usr/local && \
    ln -s /usr/local/go/bin/go /usr/local/bin/go && \
    ln -s /usr/local/go/bin/gofmt /usr/local/bin/gofmt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /go/src/github.com/ollama/ollama/
COPY . .
ARG CGO_CFLAGS
ENV GOARCH arm64
ARG VERSION
RUN --mount=type=cache,target=/root/.ccache \
    make -j 5 dist_cuda_v12 \
        CUDA_ARCHITECTURES="87" \
        GPU_RUNNER_VARIANT=_jetpack6 \
        DIST_LIB_DIR=/go/src/github.com/ollama/ollama/dist/linux-arm64-jetpack6/lib/ollama \
        DIST_GPU_RUNNER_DEPS_DIR=/go/src/github.com/ollama/ollama/dist/linux-arm64-jetpack6/lib/ollama/cuda_jetpack6

FROM --platform=linux/arm64 unified-builder-arm64 AS build-arm64
COPY . .
ARG OLLAMA_SKIP_CUDA_GENERATE
ARG OLLAMA_FAST_BUILD
ARG VERSION
RUN --mount=type=cache,target=/root/.ccache \
    make -j 5 dist
COPY --from=runners-jetpack5-arm64 /go/src/github.com/ollama/ollama/dist/ dist/
COPY --from=runners-jetpack6-arm64 /go/src/github.com/ollama/ollama/dist/ dist/
RUN cd dist/linux-$GOARCH && \
    tar -cf - . | pigz --best > ../ollama-linux-$GOARCH.tgz
RUN cd dist/linux-$GOARCH-jetpack5 && \
    tar -cf - . | pigz --best > ../ollama-linux-$GOARCH-jetpack5.tgz
RUN cd dist/linux-$GOARCH-jetpack6 && \
    tar -cf - . | pigz --best > ../ollama-linux-$GOARCH-jetpack6.tgz

FROM --platform=linux/amd64 scratch AS dist-amd64
COPY --from=build-amd64 /go/src/github.com/ollama/ollama/dist/ollama-linux-*.tgz /
FROM --platform=linux/arm64 scratch AS dist-arm64
COPY --from=build-arm64 /go/src/github.com/ollama/ollama/dist/ollama-linux-*.tgz /
FROM dist-$TARGETARCH AS dist


# For amd64 container images, filter out cuda/rocm to minimize size
FROM build-amd64 AS runners-cuda-amd64
RUN rm -rf \
    ./dist/linux-amd64/lib/ollama/libggml_hipblas.so \
    ./dist/linux-amd64/lib/ollama/runners/rocm*

FROM build-amd64 AS runners-rocm-amd64
RUN rm -rf \
    ./dist/linux-amd64/lib/ollama/libggml_cuda*.so \
    ./dist/linux-amd64/lib/ollama/libcu*.so* \
    ./dist/linux-amd64/lib/ollama/runners/cuda*

FROM --platform=linux/amd64 ubuntu:22.04 AS runtime-amd64
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=build-amd64 /go/src/github.com/ollama/ollama/dist/linux-amd64/bin/ /bin/
COPY --from=runners-cuda-amd64 /go/src/github.com/ollama/ollama/dist/linux-amd64/lib/ /lib/

FROM --platform=linux/arm64 ubuntu:22.04 AS runtime-arm64
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=build-arm64 /go/src/github.com/ollama/ollama/dist/linux-arm64/bin/ /bin/
COPY --from=build-arm64 /go/src/github.com/ollama/ollama/dist/linux-arm64/lib/ /lib/
COPY --from=runners-jetpack5-arm64 /go/src/github.com/ollama/ollama/dist/linux-arm64-jetpack5/lib/ /lib/
COPY --from=runners-jetpack6-arm64 /go/src/github.com/ollama/ollama/dist/linux-arm64-jetpack6/lib/ /lib/


# ROCm libraries larger so we keep it distinct from the CPU/CUDA image
FROM --platform=linux/amd64 ubuntu:22.04 AS runtime-rocm
# Frontload the rocm libraries which are large, and rarely change to increase chance of a common layer
# across releases
COPY --from=build-amd64 /go/src/github.com/ollama/ollama/dist/linux-amd64-rocm/lib/ /lib/
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=build-amd64 /go/src/github.com/ollama/ollama/dist/linux-amd64/bin/ /bin/
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
