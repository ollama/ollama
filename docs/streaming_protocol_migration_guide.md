# Streaming Protocol Migration Guide

## Overview

This guide explains the migration from the legacy file-based tensor transfer protocol to the new streaming protocol for cluster operations in Ollama. The streaming protocol offers significant performance, reliability, and functionality improvements for distributed model operations.

## Benefits of the Streaming Protocol

- **Improved Performance**: Chunked transfers with configurable chunk sizes for optimal throughput
- **Enhanced Reliability**: Built-in retry mechanism with exponential backoff for handling network issues
- **Compression Support**: Optional compression to reduce network bandwidth requirements
- **Better Memory Efficiency**: Streaming processing reduces memory requirements for large models
- **Real-time Progress Tracking**: Detailed transfer status and progress information
- **Resumable Transfers**: Ability to resume interrupted transfers from the last successful chunk
- **Enhanced Security**: Improved verification with checksums for data integrity

## Key Configuration Options

The streaming protocol can be configured through environment variables or the cluster configuration file:

| Setting | Environment Variable | Description | Default |
|---------|---------------------|-------------|---------|
| Enable Streaming | `OLLAMA_TENSOR_STREAMING_PROTOCOL` | Enable/disable streaming (true/false) | `true` |
| Chunk Size | `OLLAMA_TENSOR_CHUNK_SIZE` | Size of chunks in bytes | `1048576` (1MB) |
| Enable Compression | `OLLAMA_TENSOR_ENABLE_COMPRESSION` | Enable/disable compression (true/false) | `true` |
| Compression Threshold | `OLLAMA_TENSOR_COMPRESSION_THRESHOLD` | Minimum size in bytes before compression is applied | `4096` (4KB) |
| Max Retries | `OLLAMA_TENSOR_MAX_RETRIES` | Maximum number of retries for failed transfers | `3` |
| Retry Base Delay | `OLLAMA_TENSOR_RETRY_DELAY` | Base delay before retrying (milliseconds) | `500` |

## Configuration File Example

```json
{
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

## Migration Notes

### Automatic Migration

As of this release, all tensor transfers now use the streaming protocol by default. The old file-based protocol has been deprecated and is no longer used.

No action is required from users - your cluster will automatically use the new streaming protocol with sensible defaults.

### Manual Configuration

If you wish to fine-tune the protocol:

1. Set environment variables as described in the table above, or
2. Modify your cluster configuration file

### Compatibility

The streaming protocol maintains backward compatibility with older Ollama versions by detecting the capabilities of connected nodes and adjusting accordingly.

## Troubleshooting

### Performance Issues

If you experience performance issues:

- Try adjusting the chunk size (`OLLAMA_TENSOR_CHUNK_SIZE`)
  - For high-bandwidth, low-latency networks: Use larger chunks (2-4MB)
  - For high-latency networks: Use smaller chunks (256-512KB)

### High Memory Usage

If memory usage is a concern:

- Ensure compression is enabled (`OLLAMA_TENSOR_ENABLE_COMPRESSION=true`)
- Set an appropriate compression threshold (`OLLAMA_TENSOR_COMPRESSION_THRESHOLD=4096`)

### Network Timeouts

If you experience network timeouts during transfers:

- Increase max retries (`OLLAMA_TENSOR_MAX_RETRIES=5`)
- Increase retry base delay (`OLLAMA_TENSOR_RETRY_DELAY=1000`)

## Logging and Debugging

The streaming protocol provides detailed logging. Set your logging level to debug for more information:

```
export OLLAMA_LOG_LEVEL=debug
```

Look for logs with "streaming protocol" or "tensor transfer" prefixes to find relevant information.

## Technical Details

The streaming protocol works by:

1. Breaking large tensors into smaller chunks
2. Transmitting each chunk with metadata (checksums, sequence info, etc.)
3. Acknowledging successful chunk transfers
4. Handling retries automatically for failed chunks
5. Reassembling on the receiver side with verification

## Getting Help

If you encounter issues with the streaming protocol, please file an issue with:

- Description of the problem
- Log files with debug level enabled
- Your cluster configuration
- System information (OS, network setup, etc.)