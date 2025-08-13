# Ollama Server Enhancements

This document outlines the major enhancements made to the Ollama server, focusing on two key areas: **Anthropic API Compatibility** and **Enhanced Mixture of Experts (MoE) Support**.

## Overview

The enhanced Ollama server now provides:

1. **Unified Anthropic/Claude API Compatibility** - Drop-in replacement for Anthropic and Claude APIs
2. **Enhanced MoE Support** - Improved support for Mixture of Experts models with increased expert counts
3. **Backward Compatibility** - All existing Ollama functionality remains intact

## 1. Anthropic API Compatibility Layer

### Features

The Anthropic compatibility layer (`anthropic/` package) provides a unified interface that supports both Anthropic and Claude REST APIs, making Ollama a drop-in replacement for these services.

#### Key Capabilities

- **Message API Compatibility**: Full support for `/v1/messages` endpoint
- **Streaming Support**: Real-time response streaming with proper event formatting
- **Multimodal Support**: Image processing with base64 encoding (JPEG, PNG, GIF, WebP)
- **Tool Integration**: Complete tool calling support with proper argument marshaling
- **System Prompts**: Native system message handling
- **Error Handling**: Anthropic-compatible error response formatting

#### Supported Request Parameters

```json
{
  "model": "claude-3-sonnet",
  "max_tokens": 4096,
  "messages": [...],
  "system": "You are a helpful assistant",
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "stop_sequences": ["stop", "\n"],
  "stream": true,
  "tools": [...]
}
```

#### Response Format

**Non-streaming Response:**
```json
{
  "id": "msg_abc123",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Hello! How can I help you today?"
    }
  ],
  "model": "claude-3-sonnet",
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 10,
    "output_tokens": 25
  }
}
```

**Streaming Response:**
```
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text", "text": "Hello"}}

data: {"type": "message_stop", "usage": {"input_tokens": 10, "output_tokens": 25}}

data: [DONE]
```

### Implementation Details

#### Core Components

1. **Message Transformation** (`fromAnthropicMessages`)
   - Converts Anthropic message format to Ollama's internal format
   - Handles system prompts, user messages, and assistant responses
   - Processes multimodal content (text + images)

2. **Response Conversion** (`toMessageResponse`, `toMessageStreamResponse`)
   - Transforms Ollama responses to Anthropic-compatible format
   - Handles content blocks, tool calls, and usage statistics
   - Maps stop reasons between formats

3. **Middleware Integration** (`MessagesMiddleware`)
   - Validates incoming requests
   - Transforms request/response formats
   - Handles streaming and non-streaming responses

#### Content Block Processing

The system supports multiple content types:

- **Text Blocks**: Standard text responses
- **Tool Use Blocks**: Function calls with structured arguments
- **Image Blocks**: Base64-encoded images with media type validation

#### Error Handling

Comprehensive error handling with Anthropic-compatible error responses:

```json
{
  "error": {
    "type": "api_error",
    "message": "max_tokens is required and must be greater than 0"
  }
}
```

### API Endpoints

The compatibility layer adds the following endpoint:

- `POST /v1/messages` - Anthropic/Claude messages API endpoint

### Headers Support

The server accepts and processes Anthropic-specific headers:

- `anthropic-version`
- `anthropic-beta`
- `claude-version`
- `x-claude-client`

## 2. Enhanced Mixture of Experts (MoE) Support

### Overview

The enhanced MoE implementation significantly improves support for Mixture of Experts models, including increased expert counts and better performance optimization.

### Key Improvements

#### Increased Expert Limits

- **Maximum Experts**: Increased to 384 experts (supporting DeepSeekV3)
- **Dynamic Expert Selection**: Improved top-k expert selection algorithms
- **Memory Optimization**: Better memory management for large expert counts

#### Enhanced Architecture Support

The system now supports multiple MoE architectures:

- **Qwen2MoE**: Qwen2 with Mixture of Experts
- **Qwen3MoE**: Latest Qwen3 MoE variants
- **PhiMoE**: Microsoft Phi models with MoE
- **GraniteMoE**: IBM Granite MoE models
- **BailingMoE**: Bailing MoE architecture
- **NomicBertMoE**: Nomic BERT with MoE

#### Expert Gating Functions

Support for multiple expert gating mechanisms:

```c
enum llama_expert_gating_func_type {
    LLAMA_EXPERT_GATING_FUNC_TYPE_NONE    = 0,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX = 1,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID = 2
}
```

#### Performance Optimizations

1. **Quantization Support**: Improved quantization for MoE models
2. **Memory Efficiency**: Better memory allocation for expert weights
3. **Parallel Processing**: Enhanced parallel expert computation
4. **Layer-wise Processing**: Optimized layer-by-layer expert processing

### Technical Implementation

#### Core MoE Components

1. **Expert Selection** (`build_moe_ffn`)
   - Top-k expert selection with configurable k values
   - Weight normalization and scaling
   - Bias handling for expert selection

2. **Expert Processing**
   - Parallel expert computation
   - Efficient weight aggregation
   - Memory-optimized expert routing

3. **Model Parameters**
   - `n_expert`: Total number of experts
   - `n_expert_used`: Number of experts used per token
   - `n_expert_shared`: Number of shared experts
   - `expert_weights_scale`: Expert weight scaling factor

#### Quantization Enhancements

The quantization system has been enhanced to handle MoE models more effectively:

- **Expert-aware Quantization**: Different quantization strategies for different expert counts
- **Memory Optimization**: Reduced memory usage for large expert models
- **Performance Tuning**: Optimized quantization for 8-expert and larger models

## 3. Integration and Compatibility

### Backward Compatibility

All enhancements maintain full backward compatibility with existing Ollama functionality:

- Existing API endpoints remain unchanged
- Original model support is preserved
- Configuration options are additive, not breaking

### Cross-API Compatibility

The server now supports multiple API formats simultaneously:

- **Native Ollama API**: `/api/chat`, `/api/generate`
- **OpenAI Compatible API**: `/v1/chat/completions`
- **Anthropic Compatible API**: `/v1/messages`

### Model Support

Enhanced support for various model architectures:

- Traditional transformer models
- MoE models with varying expert counts
- Multimodal models (text + vision)
- Tool-calling capable models

## 4. Usage Examples

### Anthropic API Usage

```bash
# Basic chat completion
curl -X POST http://localhost:11434/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "llama3.2",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'

# Streaming response
curl -X POST http://localhost:11434/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "llama3.2",
    "max_tokens": 1024,
    "stream": true,
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ]
  }'

# With system prompt and tools
curl -X POST http://localhost:11434/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "llama3.2",
    "max_tokens": 1024,
    "system": "You are a helpful assistant with access to tools.",
    "messages": [
      {"role": "user", "content": "What is the weather like in Paris?"}
    ],
    "tools": [
      {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "input_schema": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          },
          "required": ["location"]
        }
      }
    ]
  }'
```

### MoE Model Usage

```bash
# Running MoE models (same as regular models)
ollama run qwen2.5:32b-instruct-q4_K_M

# The enhanced MoE support is automatic and transparent
# No special configuration required
```

## 5. Testing and Validation

### Test Coverage

The enhancements include comprehensive test suites:

#### Anthropic API Tests (`anthropic/anthropic_test.go`)

- Message format conversion testing
- Content block processing validation
- Tool calling functionality tests
- Error handling verification
- Streaming response testing

#### MoE Integration Tests

- Expert selection algorithm validation
- Memory usage optimization tests
- Performance benchmarking
- Quantization accuracy tests

### Validation Scenarios

1. **API Compatibility**: Verified against Anthropic API specifications
2. **Performance**: Benchmarked against baseline Ollama performance
3. **Memory Usage**: Validated memory efficiency improvements
4. **Model Accuracy**: Ensured no degradation in model output quality

## 6. Configuration and Deployment

### Environment Variables

No new environment variables are required. The enhancements work with existing Ollama configuration.

### Server Configuration

The enhanced server automatically detects and enables:

- Anthropic API compatibility when `/v1/messages` is accessed
- MoE optimizations when MoE models are loaded
- Appropriate quantization strategies based on model architecture

### Monitoring and Logging

Enhanced logging provides visibility into:

- Expert selection decisions
- API compatibility layer operations
- Performance metrics for MoE models
- Request/response transformations

## 7. Future Enhancements

### Planned Improvements

1. **Additional API Compatibility**: Support for more AI service APIs
2. **Advanced MoE Features**: Dynamic expert routing, expert specialization
3. **Performance Optimizations**: Further memory and compute optimizations
4. **Monitoring Tools**: Enhanced observability and debugging tools

### Extensibility

The architecture is designed for easy extension:

- Modular API compatibility layers
- Pluggable expert selection algorithms
- Configurable quantization strategies
- Extensible model architecture support

## 8. Conclusion

These enhancements significantly expand Ollama's capabilities while maintaining its core simplicity and performance. The Anthropic API compatibility makes Ollama a viable drop-in replacement for commercial AI services, while the enhanced MoE support enables running state-of-the-art mixture of experts models efficiently.

The implementation prioritizes:

- **Compatibility**: Drop-in replacement capability
- **Performance**: Optimized for speed and memory efficiency
- **Reliability**: Comprehensive testing and error handling
- **Maintainability**: Clean, modular architecture

These enhancements position Ollama as a comprehensive, high-performance solution for running large language models locally with commercial API compatibility.