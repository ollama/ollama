#!/bin/bash

# generate-openapi.sh

# Ollama OpenAPI Spec Generator using swaggo/swag in Docker
# This script automatically generates OpenAPI 3.0 specification from Go code annotations

set -e

echo "🚀 Generating OpenAPI specification for Ollama API using Docker..."

# Check if we're in the right directory (relative to project root)
PROJECT_ROOT=$(pwd)
if [ ! -f "$PROJECT_ROOT/go.mod" ]; then
    echo "❌ Error: Must be run from the Ollama project root (where go.mod exists)"
    echo "   Current directory: $PROJECT_ROOT"
    exit 1
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is required but not installed or not in PATH"
    echo "   Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "❌ Error: Docker daemon is not running"
    echo "   Please start Docker and try again"
    exit 1
fi

# Create docs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/docs"

echo "🔍 Running swaggo/swag in Go container..."

# Use the latest Go image with swag installed
docker run --rm \
    -v "$PROJECT_ROOT:/workspace" \
    -w /workspace \
    golang:1.24 \
    sh -c "
        echo '📦 Installing swaggo/swag...'
        go install github.com/swaggo/swag/cmd/swag@latest
        
        echo '🔍 Parsing Go code and generating OpenAPI specification...'
        /go/bin/swag init \
            --generalInfo server/routes.go \
            --dir ./ \
            --output docs/ \
            --parseDependency \
            --parseInternal \
            --propertyStrategy camelcase
    "

echo "✅ OpenAPI specification generated successfully!"
echo "📁 Files created in docs/:"
ls -la "$PROJECT_ROOT/docs/"
echo ""
echo "🌐 To view the documentation:"
echo "   - Online: Upload docs/swagger.json to https://editor.swagger.io/"
echo "   - VS Code: Install 'Swagger Viewer' extension and open docs/swagger.yaml"
echo '   - Local server: docker run --rm -p 8080:8080 -e SWAGGER_JSON=/oas/swagger.json -v ${PWD}/docs:/oas swaggerapi/swagger-ui'
