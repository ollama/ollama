FROM golang:1.20
RUN apt-get update && apt-get install -y cmake
WORKDIR /go/src/github.com/jmorganca/ollama
COPY . .
RUN go generate ./...
RUN CGO_ENABLED=1 go build -ldflags '-linkmode external -extldflags "-static"' .

FROM alpine
COPY --from=0 /go/src/github.com/jmorganca/ollama/ollama /bin/ollama
EXPOSE 80
EXPOSE 443
ARG USER=ollama
ARG GROUP=ollama
RUN addgroup -g 1000 $GROUP && adduser -u 1000 -DG $GROUP $USER
USER $USER:$GROUP
ENTRYPOINT ["/bin/ollama"]
