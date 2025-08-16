# Ollama

## Project Overview

Ollama is a Go application that allows users to run large language models locally. It provides a command-line interface (CLI) for interacting with the models, as well as a REST API for programmatic access. Ollama supports a variety of models, including Llama 2, and allows for customization of models and prompts.

The project is built using Go and has dependencies on several libraries, including:

*   **Gin:** A web framework for building the REST API.
*   **Cobra:** A library for creating powerful CLIs.
*   **Testify:** A testing toolkit for Go.

## Released Components

This codebase releases the following major components:

*   **macOS Application:** A native macOS application that provides a user-friendly interface for interacting with Ollama.
*   **Windows Application:** A native Windows application that provides a user-friendly interface for interacting with Ollama.
*   **Command-Line Interface (CLI):** The `ollama` CLI tool provides a simple interface for running and managing large language models.
*   **REST API:** The Ollama API allows for programmatic access to the models, enabling developers to build custom applications on top of Ollama.
*   **Docker Images:** Official Docker images are available on Docker Hub, making it easy to deploy Ollama in a containerized environment.
*   **Python and JavaScript Libraries:** Ollama provides official Python and JavaScript libraries for interacting with the REST API.

## Building and Running

### Prerequisites

*   Go
*   C/C++ Compiler (Clang on macOS, TDM-GCC or llvm-mingw on Windows, GCC/Clang on Linux)
*   CMake (for some platforms)

### Building

The project can be built using the `go run` or `go build` commands. For specific platform instructions, refer to the [development documentation](docs/development.md).

### Running

To run the Ollama server, use the following command:

```shell
go run . serve
```

In a separate shell, you can then run a model:

```shell
ollama run llama3.2
```

### Testing

To run the project's tests, use the following command:

```shell
go test ./...
```

## Development Conventions

The project follows standard Go development conventions. All code is formatted using `gofmt`. The project also uses `golangci-lint` for linting.
