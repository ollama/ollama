FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG TARGETARCH
ARG VERSION=0.0.0
ARG GOFLAGS="'-ldflags=-w -s'"

WORKDIR /go/src/github.com/jmorganca/ollama
RUN apt-get update && apt-get install -y git build-essential cmake
ADD https://dl.google.com/go/go1.21.1.linux-$TARGETARCH.tar.gz /tmp/go1.21.1.tar.gz
RUN mkdir -p /usr/local && tar xz -C /usr/local </tmp/go1.21.1.tar.gz

COPY . .
ENV GOARCH=$TARGETARCH
RUN /usr/local/go/bin/go generate ./... \
    && /usr/local/go/bin/go build .

FROM ubuntu:22.04
RUN apt-get update && apt-get install -y ca-certificates
COPY --from=0 /go/src/github.com/jmorganca/ollama/ollama /bin/ollama
EXPOSE 11434
ENV OLLAMA_HOST 0.0.0.0
ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]
