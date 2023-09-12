#!/bin/bash

# Define a function to print in red color
print_red() {
    echo -e "\033[31m$1\033[0m"
}

# Check if 'go' is installed
which go >/dev/null 2>&1
if [ $? -ne 0 ]; then
    print_red "Error: 'go' is not installed."
    exit 1
fi

# Check if 'cmake' is installed
which cmake >/dev/null 2>&1
if [ $? -ne 0 ]; then
    print_red "Error: 'cmake' is not installed."
    exit 1
fi

# If both are installed, build ollama:
go generate ./... && \
go build -ldflags '-linkmode external -extldflags "-static"' .

if [ $? -ne 0 ]; then
    print_red "Error: Failed to build ollama."
    exit 1
fi

echo "Successfully build ollama!"
