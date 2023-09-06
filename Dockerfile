FROM golang:alpine

WORKDIR /go/src/github.com/jmorganca/ollama
RUN apk add --no-cache git build-base cmake

COPY . .
RUN go generate ./... && go build -ldflags '-linkmode external -extldflags "-static"' .

FROM alpine
ENV OLLAMA_HOST 0.0.0.0
RUN apk add --no-cache libstdc++

ARG USER=ollama
ARG GROUP=ollama
RUN addgroup $GROUP && adduser -D -G $GROUP $USER

COPY --from=0 /go/src/github.com/jmorganca/ollama/ollama /bin/ollama

USER $USER:$GROUP
ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]
