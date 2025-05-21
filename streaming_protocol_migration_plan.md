# Streaming Protocol Migration Guide

## Overview

This document outlines the migration from the original file-based tensor transfer protocol to the new streaming protocol in Ollama's cluster mode. The streaming protocol provides several advantages:

- **Chunked transfers**: Tensors are broken into smaller chunks for more reliable transfers
- **Compression**: Optional data compression to reduce bandwidth requirements
- **Resumable transfers**: Ability to resume interrupted transfers from checkpoints
- **Better error handling**: Improved retry mechanisms and error reporting
- **Prioritization**: Queue management based on priority levels
- **Progress tracking**: Better visibility into transfer progress

## Configuration Options

The new streaming protocol can be configured via the `TensorProtocol` section in your cluster configuration:

```json
{
  "enabled": true,
  "node_name": "worker-node-1",
  "node_role": "worker",
  "tensor_protocol": {
    "use_streaming_protocol": true,
    "chunk_size": 1048576,
    "enable_compression": true,
    "compression_threshold": 4096,
    "max_retries": 3,
    "retry_base_delay": 500
  }
}
```

### Available Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `use_streaming_protocol` | Enable/disable streaming protocol | `true` |
| `chunk_size` | Size of chunks in bytes | `1048576` (1MB) |
| `enable_compression` | Enable data compression | `true` |
| `compression_threshold` | Minimum size in bytes for compression | `4096` (4KB) |
| `max_retries` | Maximum retry attempts | `3` |
| `retry_base_delay` | Base delay in ms before retrying | `500` |

### Environment Variables

These settings can also be configured via environment variables:

- `OLLAMA_TENSOR_STREAMING_PROTOCOL`: Set to "true" to enable streaming protocol
- `OLLAMA_TENSOR_CHUNK_SIZE`: Chunk size in bytes
- `OLLAMA_TENSOR_ENABLE_COMPRESSION`: Set to "true" to enable compression
- `OLLAMA_TENSOR_COMPRESSION_THRESHOLD`: Minimum size for compression in bytes
- `OLLAMA_TENSOR_MAX_RETRIES`: Maximum retry attempts
- `OLLAMA_TENSOR_RETRY_DELAY`: Base retry delay in milliseconds

## Code Changes

If you're developing extensions or integrations with Ollama cluster mode, note these API changes:

### Creating a protocol instance
```go
// Old approach - standard protocol
protocol := tensor.NewProtocol(conn)

// New approach - streaming protocol
protocol := tensor.NewStreamingProtocol(conn)

// Configure streaming options
protocol.SetChunkSize(1024 * 1024)  // 1MB chunks
protocol.SetCompressionEnabled(true)
protocol.SetRetryPolicy(tensor.RetryPolicy{
    MaxRetries:    3,
    BaseDelay:     500 * time.Millisecond,
    MaxDelay:      5 * time.Second,
    BackoffFactor: 2.0,
    JitterFactor:  0.1,
})
```

### Sending data
```go
// Old approach - standard protocol
err := protocol.SendTensorSync(modelID, partitionID, tensorID, data)

// New approach - streaming protocol
err := protocol.StreamTensor(modelID, partitionID, tensorID, data)
```

### Requesting data
```go
// Old approach - standard protocol
err := protocol.SendTensorRequest(modelID, partitionID, tensorID)
header, data, err := protocol.ReceiveMessage()

// New approach - streaming protocol
streamID, err := protocol.RequestTensor(modelID, partitionID, tensorID, priority)
// Handle incoming chunks in a loop until complete
```

## Backward Compatibility

The streaming protocol maintains backward compatibility with the standard protocol through adapter methods. If a component uses the standard protocol methods, they will be automatically routed through the streaming protocol with default settings.

## Migration Steps

1. Update your cluster configuration to include `tensor_protocol` settings
2. If you have custom code that interfaces with the tensor protocol:
   - Update calls to use streaming protocol methods
   - Configure chunk size and compression based on your needs
3. For custom integrations, update error handling to account for chunked transfers

## Troubleshooting

- If transfers are failing, try increasing `max_retries` and `retry_base_delay`
- For large models, increase `chunk_size` to reduce overhead
- For slow networks, enable compression and reduce `chunk_size`
- Monitor transfer progress through debug logs or the cluster status API