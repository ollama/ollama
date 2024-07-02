ARG GOLANG_VERSION=1.22.1
ARG CMAKE_VERSION=3.22.1
# this CUDA_VERSION corresponds with the one specified in docs/gpu.md
ARG CUDA_VERSION=11.3.1
ARG ROCM_VERSION=6.1.1
ARG JETPACK_6=r36.2.0
ARG JETPACK_5=r35.4.1
ARG JETPACK_4=r32.7.1

# Copy the minimal context we need to run the generate scripts
FROM scratch AS llm-code
COPY .git .git
COPY .gitmodules .gitmodules
COPY llm llm

FROM --platform=linux/amd64 nvidia/cuda:$CUDA_VERSION-devel-centos7 AS cuda-build-amd64
ARG CMAKE_VERSION
COPY ./scripts/rh_linux_deps.sh /
RUN CMAKE_VERSION=${CMAKE_VERSION} sh /rh_linux_deps.sh
ENV PATH /opt/rh/devtoolset-10/root/usr/bin:$PATH
COPY --from=llm-code / /go/src/github.com/ollama/ollama/
WORKDIR /go/src/github.com/ollama/ollama/llm/generate
ARG CGO_CFLAGS
RUN OLLAMA_SKIP_STATIC_GENERATE=1 OLLAMA_SKIP_CPU_GENERATE=1 bash gen_linux.sh

FROM --platform=linux/arm64 nvidia/cuda:$CUDA_VERSION-devel-rockylinux8 AS cuda-build-server-arm64
ARG CMAKE_VERSION
COPY ./scripts/rh_linux_deps.sh /
RUN CMAKE_VERSION=${CMAKE_VERSION} sh /rh_linux_deps.sh
ENV PATH /opt/rh/gcc-toolset-10/root/usr/bin:$PATH
COPY --from=llm-code / /go/src/github.com/ollama/ollama/
WORKDIR /go/src/github.com/ollama/ollama/llm/generate
ARG CGO_CFLAGS
RUN OLLAMA_SKIP_STATIC_GENERATE=1 OLLAMA_SKIP_CPU_GENERATE=1 bash gen_linux.sh

FROM --platform=linux/arm64 nvcr.io/nvidia/l4t-jetpack:${JETPACK_6} AS cuda-build-jetpack6-arm64
ARG CMAKE_VERSION
RUN apt-get update && apt-get install -y git curl && \
    curl -s -L https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-$(uname -m).tar.gz | tar -zx -C /usr --strip-components 1
COPY --from=llm-code / /go/src/github.com/ollama/ollama/
WORKDIR /go/src/github.com/ollama/ollama/llm/generate
ARG CGO_CFLAGS
RUN OLLAMA_SKIP_STATIC_GENERATE=1 \
    OLLAMA_SKIP_CPU_GENERATE=1 \
    CUDA_VARIANT="_jetpack6" \
    CMAKE_CUDA_ARCHITECTURES="87" \
    bash gen_linux.sh

FROM --platform=linux/arm64 nvcr.io/nvidia/l4t-jetpack:${JETPACK_5} AS cuda-build-jetpack5-arm64
ARG CMAKE_VERSION
RUN apt-get update && apt-get install -y git curl && \
    curl -s -L https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-$(uname -m).tar.gz | tar -zx -C /usr --strip-components 1
COPY --from=llm-code / /go/src/github.com/ollama/ollama/
WORKDIR /go/src/github.com/ollama/ollama/llm/generate
ARG CGO_CFLAGS
RUN OLLAMA_SKIP_STATIC_GENERATE=1 \
    OLLAMA_SKIP_CPU_GENERATE=1 \
    CUDA_VARIANT="_jetpack5" \
    CMAKE_CUDA_ARCHITECTURES="72;87" \
    bash gen_linux.sh

# TODO - this is still broken - 
# 1.114 -- Could NOT find CUDAToolkit (missing: CUDA_CUDART) (found version "10.2.300")
# 1.114 CMake Warning at CMakeLists.txt:463 (message):
# 1.114   CUDA not found
# FROM --platform=linux/arm64 nvcr.io/nvidia/l4t-base:${JETPACK_4} AS cuda-build-jetpack4-arm64
# ARG CMAKE_VERSION
# RUN apt update && \
#     apt-get install -y software-properties-common && \
#     add-apt-repository ppa:ubuntu-toolchain-r/test &&\
#     apt-get update && \
#     apt-get update && apt-get install -y git curl make gcc-10 g++-10 && \
#     update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 30 && \
#     update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 30 && \
#     curl -s -L https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-$(uname -m).tar.gz | tar -zx -C /usr --strip-components 1
# COPY --from=llm-code / /go/src/github.com/ollama/ollama/
# WORKDIR /go/src/github.com/ollama/ollama/llm/generate
# ARG CGO_CFLAGS
# RUN OLLAMA_SKIP_STATIC_GENERATE=1 \
#     OLLAMA_SKIP_CPU_GENERATE=1 \
#     CUDA_VARIANT="_jetpack4" \
#     CMAKE_CUDA_ARCHITECTURES="53;72" \
#     bash gen_linux.sh

FROM --platform=linux/amd64 rocm/dev-centos-7:${ROCM_VERSION}-complete AS rocm-build-amd64
ARG CMAKE_VERSION
COPY ./scripts/rh_linux_deps.sh /
RUN CMAKE_VERSION=${CMAKE_VERSION} sh /rh_linux_deps.sh
ENV PATH /opt/rh/devtoolset-10/root/usr/bin:$PATH
ENV LIBRARY_PATH /opt/amdgpu/lib64
COPY --from=llm-code / /go/src/github.com/ollama/ollama/
WORKDIR /go/src/github.com/ollama/ollama/llm/generate
ARG CGO_CFLAGS
ARG AMDGPU_TARGETS
RUN OLLAMA_SKIP_STATIC_GENERATE=1 OLLAMA_SKIP_CPU_GENERATE=1 bash gen_linux.sh
RUN mkdir /tmp/scratch && \
    for dep in $(zcat /go/src/github.com/ollama/ollama/llm/build/linux/x86_64/rocm*/bin/deps.txt.gz) ; do \
        cp ${dep} /tmp/scratch/ || exit 1 ; \
    done && \
    (cd /opt/rocm/lib && tar cf - rocblas/library) | (cd /tmp/scratch/ && tar xf - ) && \
    mkdir -p /go/src/github.com/ollama/ollama/dist/deps/ && \
    (cd /tmp/scratch/ && tar czvf /go/src/github.com/ollama/ollama/dist/deps/ollama-linux-amd64-rocm.tgz . )


FROM --platform=linux/amd64 centos:7 AS cpu-builder-amd64
ARG CMAKE_VERSION
ARG GOLANG_VERSION
COPY ./scripts/rh_linux_deps.sh /
RUN CMAKE_VERSION=${CMAKE_VERSION} GOLANG_VERSION=${GOLANG_VERSION} sh /rh_linux_deps.sh
ENV PATH /opt/rh/devtoolset-10/root/usr/bin:$PATH
COPY --from=llm-code / /go/src/github.com/ollama/ollama/
ARG OLLAMA_CUSTOM_CPU_DEFS
ARG CGO_CFLAGS
WORKDIR /go/src/github.com/ollama/ollama/llm/generate

FROM --platform=linux/amd64 cpu-builder-amd64 AS static-build-amd64
RUN OLLAMA_CPU_TARGET="static" bash gen_linux.sh
FROM --platform=linux/amd64 cpu-builder-amd64 AS cpu-build-amd64
RUN OLLAMA_SKIP_STATIC_GENERATE=1 OLLAMA_CPU_TARGET="cpu" bash gen_linux.sh
FROM --platform=linux/amd64 cpu-builder-amd64 AS cpu_avx-build-amd64
RUN OLLAMA_SKIP_STATIC_GENERATE=1 OLLAMA_CPU_TARGET="cpu_avx" bash gen_linux.sh
FROM --platform=linux/amd64 cpu-builder-amd64 AS cpu_avx2-build-amd64
RUN OLLAMA_SKIP_STATIC_GENERATE=1 OLLAMA_CPU_TARGET="cpu_avx2" bash gen_linux.sh

FROM --platform=linux/arm64 centos:7 AS cpu-builder-arm64
ARG CMAKE_VERSION
ARG GOLANG_VERSION
COPY ./scripts/rh_linux_deps.sh /
RUN CMAKE_VERSION=${CMAKE_VERSION} GOLANG_VERSION=${GOLANG_VERSION} sh /rh_linux_deps.sh
ENV PATH /opt/rh/devtoolset-10/root/usr/bin:$PATH
COPY --from=llm-code / /go/src/github.com/ollama/ollama/
ARG OLLAMA_CUSTOM_CPU_DEFS
ARG CGO_CFLAGS
WORKDIR /go/src/github.com/ollama/ollama/llm/generate

FROM --platform=linux/arm64 cpu-builder-arm64 AS static-build-arm64
RUN OLLAMA_CPU_TARGET="static" bash gen_linux.sh
FROM --platform=linux/arm64 cpu-builder-arm64 AS cpu-build-arm64
RUN OLLAMA_SKIP_STATIC_GENERATE=1 OLLAMA_CPU_TARGET="cpu" bash gen_linux.sh


# Intermediate stage used for ./scripts/build_linux.sh
FROM --platform=linux/amd64 cpu-build-amd64 AS build-amd64
ENV CGO_ENABLED 1
WORKDIR /go/src/github.com/ollama/ollama
COPY . .
COPY --from=static-build-amd64 /go/src/github.com/ollama/ollama/llm/build/linux/ llm/build/linux/
COPY --from=cpu_avx-build-amd64 /go/src/github.com/ollama/ollama/llm/build/linux/ llm/build/linux/
COPY --from=cpu_avx2-build-amd64 /go/src/github.com/ollama/ollama/llm/build/linux/ llm/build/linux/
COPY --from=cuda-build-amd64 /go/src/github.com/ollama/ollama/llm/build/linux/ llm/build/linux/
COPY --from=rocm-build-amd64 /go/src/github.com/ollama/ollama/llm/build/linux/ llm/build/linux/
COPY --from=rocm-build-amd64 /go/src/github.com/ollama/ollama/dist/deps/ ./dist/deps/
ARG GOFLAGS
ARG CGO_CFLAGS
RUN go build -trimpath .

# Intermediate stage used for ./scripts/build_linux.sh
FROM --platform=linux/arm64 cpu-build-arm64 AS build-arm64
ENV CGO_ENABLED 1
ARG GOLANG_VERSION
WORKDIR /go/src/github.com/ollama/ollama
COPY . .
COPY --from=static-build-arm64 /go/src/github.com/ollama/ollama/llm/build/linux/ llm/build/linux/
COPY --from=cuda-build-server-arm64 /go/src/github.com/ollama/ollama/llm/build/linux/ llm/build/linux/
## arm binary += 381M
COPY --from=cuda-build-jetpack6-arm64 /go/src/github.com/ollama/ollama/llm/build/linux/ llm/build/linux/
## arm binary += 330M
COPY --from=cuda-build-jetpack5-arm64 /go/src/github.com/ollama/ollama/llm/build/linux/ llm/build/linux/
## not working yet...
# COPY --from=cuda-build-jetpack4-arm64 /go/src/github.com/ollama/ollama/llm/build/linux/ llm/build/linux/
ARG GOFLAGS
ARG CGO_CFLAGS
RUN go build -trimpath .

# Runtime stages
FROM --platform=linux/amd64 ubuntu:22.04 as runtime-amd64
RUN apt-get update && apt-get install -y ca-certificates
COPY --from=build-amd64 /go/src/github.com/ollama/ollama/ollama /bin/ollama
FROM --platform=linux/arm64 ubuntu:22.04 as runtime-arm64
RUN apt-get update && apt-get install -y ca-certificates
COPY --from=build-arm64 /go/src/github.com/ollama/ollama/ollama /bin/ollama

# Radeon images are much larger so we keep it distinct from the CPU/CUDA image
FROM --platform=linux/amd64 rocm/dev-centos-7:${ROCM_VERSION}-complete as runtime-rocm
RUN update-pciids
COPY --from=build-amd64 /go/src/github.com/ollama/ollama/ollama /bin/ollama
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
