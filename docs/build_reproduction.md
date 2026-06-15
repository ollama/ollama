# Ollama s390x Build Reproduction Guide

This guide provides step-by-step instructions for building and running Ollama on s390x architecture in a Jupyter notebook environment.

## Prerequisites

- Access to an s390x system (e.g., triframe)
- SSH access configured
- Podman or Docker with compose support
- Basic familiarity with Linux command line

## Server-Side Setup

### 1. Connect to the s390x System

```bash
ssh triframe
```

### 2. Navigate to Notebooks Directory

```bash
cd notebooks
```

### 3. Start Jupyter Environment with Podman Compose

```bash
podman compose up -d
```

## Client-Side Setup

### 1. Establish SSH Tunnel

Set up an SSH tunnel to access the Jupyter notebook from your local machine:

```bash
ssh -L 8888:localhost:8888 triframe
```

### 2. Navigate to Notebooks Directory

```bash
cd notebooks
```

### 3. Access the Ollama Container

Execute commands inside the running ollama container:

```bash
podman exec -it ollama bash
# or
docker exec -it ollama bash
```

## Build Environment Setup

### 1. Install System Dependencies

```bash
apt-get update
apt-get install -y git make golang-go build-essential pkg-config wget
```

### 2. Install Go 1.23.4 for s390x

Download and install the correct Go version for s390x architecture:

```bash
cd /tmp
wget https://go.dev/dl/go1.23.4.linux-s390x.tar.gz
rm -rf /usr/local/go
tar -C /usr/local -xzf go1.23.4.linux-s390x.tar.gz
```

### 3. Configure Go Environment

Add Go to your PATH:

```bash
export PATH=$PATH:/usr/local/go/bin
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin
```

Verify the installation:

```bash
go version
# Expected output: go version go1.23.4 linux/s390x
```

## Building Ollama

### 1. Clone the Repository

```bash
cd ~
git clone https://github.com/yourusername/ollama-s390x.git
cd ollama-s390x
```

### 2. Full Build from Repository Root

```bash
cmake -B build .
cmake --build build --parallel 8
```

### 3. Alternative: Quick Go-Only Build

For rapid iteration against an existing native payload:

```bash
go build .
```

## Running Ollama

### 1. Start the Ollama Server

```bash
./ollama serve
```

The server will start on the default port (11434).

### 2. Verify Server is Running

In a new terminal or shell session:

```bash
curl http://localhost:11434/api/version
```

## Working with Models

### 1. Pull a Model

```bash
./ollama pull llama2
```

### 2. Run a Model

```bash
./ollama run llama2
```

### 3. Test with a Prompt

```bash
./ollama run llama2 "Hello, how are you?"
```

## Notes

- The build process may take significant time on s390x architecture
- Ensure sufficient disk space is available (at least 10GB recommended)
- For development iteration, use the Go-only build method after the initial full build
- Check `docs/development.md` for additional platform-specific notes and GPU backend information

## Troubleshooting

If you encounter issues:

1. Verify all dependencies are installed correctly
2. Check that Go version matches the required version (1.23.4)
3. Ensure the PATH environment variables are set correctly
4. Review build logs for specific error messages
5. Consult `logs/build_attempt_003.md` for common errors and solutions

## References

- [Development Documentation](development.md)
- [Build Failures Log](build_failures.md)
- [s390x Architecture Notes](s390x_architecture_notes.md)