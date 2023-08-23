# Development

Install required tools:

```
brew install go cmake
```

You will also need a C/C++ compiler such as GCC for MacOS and Linux or Mingw-w64 GCC for Windows.

Get the required libraries:

```
git submodule update --init --recursive
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
