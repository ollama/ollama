# Development

- Install cmake or (optionally, required tools for GPUs)
- run `go generate ./...`
- run `go build .`

Install required tools:

- cmake version 3.24 or higher
- go version 1.20 or higher
- gcc version 11.4.0 or higher

```
brew install go cmake gcc
```

Get the required libraries:

```
go generate ./...
```

Then build ollama:

```
go build .
```

Now you can run `ollama`:

```
./ollama
```

## Building on Linux with GPU support

- Install cmake and nvidia-cuda-toolkit
- run `go generate ./...`
- run `go build .`
