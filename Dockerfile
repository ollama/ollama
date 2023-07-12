FROM golang:1.20
WORKDIR /go/src/github.com/jmorganca/ollama
COPY . .
RUN CGO_ENABLED=1 go build -ldflags '-linkmode external -extldflags "-static"' .

FROM alpine
COPY --from=0 /go/src/github.com/jmorganca/ollama/ollama /bin/ollama
EXPOSE 11434
ARG USER=ollama
ARG GROUP=ollama
RUN addgroup -g 1000 $GROUP && adduser -u 1000 -DG $GROUP $USER
USER $USER:$GROUP
ENTRYPOINT ["/bin/ollama"]
ENV OLLAMA_HOST 0.0.0.0
CMD ["serve"]
