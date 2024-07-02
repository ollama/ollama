ARG GOLANG_VERSION=1.22.1
ARG CMAKE_VERSION=3.22.1
# this CUDA_VERSION corresponds with the one specified in docs/gpu.md
ARG CUDA_VERSION=11.3.1
ARG ROCM_VERSION=6.1.1

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
RUN OLLAMA_SKIP_STATIC_GENERATE=1 OLLAMA_SKIP_CPU_GENERATE=1 sh gen_linux.sh

FROM --platform=linux/arm64 nvidia/cuda:$CUDA_VERSION-devel-rockylinux8 AS cuda-build-arm64
ARG CMAKE_VERSION
COPY ./scripts/rh_linux_deps.sh /
RUN CMAKE_VERSION=${CMAKE_VERSION} sh /rh_linux_deps.sh
ENV PATH /opt/rh/gcc-toolset-10/root/usr/bin:$PATH
COPY --from=llm-code / /go/src/github.com/ollama/ollama/
WORKDIR /go/src/github.com/ollama/ollama/llm/generate
ARG CGO_CFLAGS
RUN OLLAMA_SKIP_STATIC_GENERATE=1 OLLAMA_SKIP_CPU_GENERATE=1 sh gen_linux.sh

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
RUN OLLAMA_SKIP_STATIC_GENERATE=1 OLLAMA_SKIP_CPU_GENERATE=1 sh gen_linux.sh
RUN mkdir /tmp/scratch && \
    for dep in $(zcat /go/src/github.com/ollama/ollama/llm/build/linux/x86_64/rocm*/bin/deps.txt.gz) ; do \
        cp ${dep} /tmp/scratch/ || exit 1 ; \
    done && \
    (cd /opt/rocm/lib && tar cf - rocblas/library) | (cd /tmp/scratch/ && tar xf - ) && \
    mkdir -p /go/src/github.com/ollama/ollama/dist/deps/ && \
    (cd /tmp/scratch/ && tar czvf /go/src/github.com/ollama/ollama/dist/deps/ollama-linux-amd64-rocm.tgz . )


FROM --platform=linux/amd64 intel/oneapi-basekit:2024.1.0-devel-rockylinux9 AS oneapi-build-amd64
ARG CMAKE_VERSION
COPY ./scripts/rh_linux_deps.sh /
RUN CMAKE_VERSION=${CMAKE_VERSION} sh /rh_linux_deps.sh
COPY --from=llm-code / /go/src/github.com/ollama/ollama/
WORKDIR /go/src/github.com/ollama/ollama/llm/generate
ARG CGO_CFLAGS
RUN OLLAMA_SKIP_STATIC_GENERATE=1 OLLAMA_SKIP_CPU_GENERATE=1 sh gen_linux.sh

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
RUN OLLAMA_CPU_TARGET="static" sh gen_linux.sh
FROM --platform=linux/amd64 cpu-builder-amd64 AS cpu-build-amd64
RUN OLLAMA_SKIP_STATIC_GENERATE=1 OLLAMA_CPU_TARGET="cpu" sh gen_linux.sh
FROM --platform=linux/amd64 cpu-builder-amd64 AS cpu_avx-build-amd64
RUN OLLAMA_SKIP_STATIC_GENERATE=1 OLLAMA_CPU_TARGET="cpu_avx" sh gen_linux.sh
FROM --platform=linux/amd64 cpu-builder-amd64 AS cpu_avx2-build-amd64
RUN OLLAMA_SKIP_STATIC_GENERATE=1 OLLAMA_CPU_TARGET="cpu_avx2" sh gen_linux.sh

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
RUN OLLAMA_CPU_TARGET="static" sh gen_linux.sh
FROM --platform=linux/arm64 cpu-builder-arm64 AS cpu-build-arm64
RUN OLLAMA_SKIP_STATIC_GENERATE=1 OLLAMA_CPU_TARGET="cpu" sh gen_linux.sh


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
COPY --from=oneapi-build-amd64 /go/src/github.com/ollama/ollama/llm/build/linux/ llm/build/linux/
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
COPY --from=cuda-build-arm64 /go/src/github.com/ollama/ollama/llm/build/linux/ llm/build/linux/
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

# use ubuntu oneapi-basekit image as runtime image, need to use --device param to mount GPU to container.
# e.g. docker run -it  -v "$(pwd):/app:Z" --device /dev/dri/renderD128:/dev/dri/renderD129 --device /dev/dri/card1:/dev/dri/card1 IMAGE_ID
# host machine need to install Intel GPU driver correctly.
# to detect the device info which you want to mount, run sudo intel_gpu_top -L
FROM --platform=linux/amd64 intel/oneapi-basekit:2024.1.0-devel-ubuntu22.04 as runtime-oneapi
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
