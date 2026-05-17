# syntax=docker/dockerfile:1
# vim: filetype=dockerfile

FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS build

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build ccache ca-certificates curl gcc g++
ENV CMAKE_GENERATOR=Ninja
ENV CMAKE_C_COMPILER_LAUNCHER=ccache
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache

WORKDIR /build
COPY CMakeLists.txt CMakePresets.json ./
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml

COPY scripts/cicc-cache /usr/local/cuda/nvvm/bin/cicc-cache
RUN chmod +x /usr/local/cuda/nvvm/bin/cicc-cache \
    && mv /usr/local/cuda/nvvm/bin/cicc /usr/local/cuda/nvvm/bin/cicc.real \
    && mv /usr/local/cuda/nvvm/bin/cicc-cache /usr/local/cuda/nvvm/bin/cicc

RUN cmake --preset 'CUDA 13' -DCMAKE_CUDA_ARCHITECTURES=86 \
    && cmake --build --preset 'CUDA 13' -j$(nproc) \
    && cmake --install build --component CUDA --strip

RUN --mount=type=cache,target=/root/.cache/ccache \
    cmake --preset CPU \
    && cmake --build --preset CPU -j$(nproc) \
    && cmake --install build --component CPU --strip

ARG GO_VERSION=1.26.0
ADD --unpack "https://golang.org/dl/go${GO_VERSION}.linux-amd64.tar.gz" /usr/local/
ENV PATH=/usr/local/go/bin:$PATH

WORKDIR /build/ollama
COPY go.mod go.sum ./

RUN --mount=type=cache,target=/root/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    go mod download
COPY . .
ENV CGO_ENABLED=1
RUN --mount=type=cache,target=/root/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    go build -trimpath -buildmode=pie -ldflags='-w -s' -o /bin/ollama .

FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update && apt-get install -y --no-install-recommends ca-certificates curl

COPY --from=build /bin/ollama /usr/bin/ollama
COPY --from=build /build/dist/lib/ollama /usr/lib/ollama

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all
ENV OLLAMA_HOST=0.0.0.0:11434
EXPOSE 11434
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:11434/health || exit 1
ENTRYPOINT ["/usr/bin/ollama"]
CMD ["serve"]
