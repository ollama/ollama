# Development

Install required tools:

```
brew install go
brew install cmake
```

Enable CGO:

```
export CGO_ENABLED=1
```

You will also need a C/C++ compiler such as GCC for MacOS and Linux or Mingw-w64 GCC for Windows.

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
