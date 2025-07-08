# OpenAPI Specification Generator for Ollama

This tool generates OpenAPI 3.0 specifications from the Ollama Go codebase using [swaggo/swag](https://github.com/swaggo/swag), the industry standard for Go API documentation.

## Why swaggo/swag?

- **Annotation-based**: Uses Go comments to generate specs - no hardcoded API knowledge
- **Truly dynamic**: Parses actual Go code structure and types
- **Industry standard**: Used by thousands of Go projects
- **Framework agnostic**: Works with any Go HTTP framework
- **Rich features**: Supports complex types, validation, examples, security, etc.

## Quick Start

```bash
# From the Ollama project root
./scripts/generate-openapi.sh
```

## Output Files

The generator creates:

- `docs/docs.go` - Go embeddable documentation
- `docs/swagger.json` - OpenAPI 3.0 JSON specification
- `docs/swagger.yaml` - OpenAPI 3.0 YAML specification

## Viewing the Documentation

1. **Local**: `docker run --rm -p 8080:8080 -e SWAGGER_JSON=/oas/swagger.json -v ${PWD}/docs:/oas swaggerapi/swagger-ui`
2. **Online**: Upload `docs/swagger.json` to [Swagger Editor](https://editor.swagger.io/)
3. **VS Code**: Install the "Swagger Viewer" extension and open `docs/swagger.yaml`

## How It Works

swaggo/swag parses Go source code and extracts:

1. **API Information**: From comments in route files
2. **Request/Response Types**: From Go struct definitions with JSON tags
3. **Route Definitions**: From HTTP handler registrations
4. **Documentation**: From Go comments above functions and types

## Adding Documentation to Ollama

To improve the generated documentation, add swag annotations to the Ollama codebase:

### General API Info (in server/routes.go or main.go)

```go
// @title           Ollama API
// @version         1.0
// @description     API for running and managing large language models locally
// @termsOfService  https://ollama.com/terms
// @contact.name    Ollama Support
// @contact.url     https://ollama.com/support
// @contact.email   support@ollama.com
// @license.name    MIT
// @license.url     https://github.com/ollama/ollama/blob/main/LICENSE
// @host            localhost:11434
// @BasePath        /
```

### API Operations (above handler functions)

```go
// ListModels godoc
// @Summary      List available models
// @Description  Get a list of all available models
// @Tags         models
// @Accept       json
// @Produce      json
// @Success      200  {object}  api.ListResponse
// @Failure      500  {object}  api.ErrorResponse
// @Router       /api/tags [get]
func ListModelsHandler(c *gin.Context) {
    // ... handler code
}
```

### Path Parameters (Important)

When using URL path parameters (like `/v1/models/{model}`), you must declare the path parameter with a `@Param` annotation:

```go
// @Param        model path string true "Model name parameter" 
// @Router       /api/models/{model} [get]
```

Failing to declare path parameters will result in OpenAPI validation errors.

### Type Documentation (above struct definitions)

```go
// GenerateRequest represents a text generation request
type GenerateRequest struct {
    Model    string `json:"model" example:"llama2"`              // The model to use for generation
    Prompt   string `json:"prompt" example:"Why is the sky blue?"` // The prompt to generate from
    Stream   bool   `json:"stream,omitempty"`                   // Whether to stream the response
}
```

## Benefits Over Custom Generators

1. **No maintenance**: swaggo/swag is actively maintained
2. **Rich features**: Supports all OpenAPI features out of the box
3. **Community standard**: Familiar to Go developers
4. **Self-documenting**: Code and docs stay in sync automatically
5. **Framework integration**: Can integrate with Gin for live docs

## Troubleshooting

If generation fails:

1. Ensure you're running from the Ollama project root
2. Check that Docker is running
3. Verify the Go code compiles without errors
4. Check swaggo/swag documentation for annotation syntax

## Further Reading

- [swaggo/swag Documentation](https://github.com/swaggo/swag)
- [OpenAPI 3.0 Specification](https://swagger.io/specification/)
- [Swagger Annotation Guide](https://github.com/swaggo/swag#declarative-comments-format)
