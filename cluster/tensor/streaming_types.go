package tensor

import (
	"fmt"
	"strings"
	"time"
)

// PriorityLevel defines the priority of tensor requests
type PriorityLevel uint8

const (
	// PriorityLow is used for background operations
	PriorityLow PriorityLevel = 0

	// PriorityNormal is the default priority
	PriorityNormal PriorityLevel = 50

	// PriorityHigh is used for foreground operations
	PriorityHigh PriorityLevel = 100

	// PriorityCritical is used for urgent operations
	PriorityCritical PriorityLevel = 200
)

// CompressionType defines the type of compression to use
type CompressionType uint8

const (
	// CompressionNone uses no compression
	CompressionNone CompressionType = 0

	// CompressionLZ4 uses LZ4 compression
	CompressionLZ4 CompressionType = 1

	// CompressionDeflate uses Deflate compression
	CompressionDeflate CompressionType = 2
)

// TransferMode defines the mode of tensor transfer
type TransferMode string

const (
	// TransferModeStreaming uses streaming protocol
	TransferModeStreaming = "streaming"

	// TransferModeStandard is the legacy non-streaming transfer mode
	// DEPRECATED: This mode is deprecated and will be removed in a future version.
	// All new code should use TransferModeStreaming instead.
	TransferModeStandard = "standard" // Deprecated: Use TransferModeStreaming

	// TransferModeAdaptive automatically chooses the best protocol
	TransferModeAdaptive = "adaptive"
)

// StreamingMessageType defines message types for the streaming protocol
type StreamingMessageType uint8

const (
	// Original message types (for compatibility)
	TypeTensorRequest  StreamingMessageType = 1
	TypeTensorResponse StreamingMessageType = 2
	TypeTensorError    StreamingMessageType = 3

	// New message types for streaming protocol
	TypeTensorStreamRequest     StreamingMessageType = 10
	TypeTensorStreamChunk       StreamingMessageType = 11
	TypeTensorStreamComplete    StreamingMessageType = 12
	TypeTensorPrefetchRequest   StreamingMessageType = 13
	TypeTensorPrefetchPriority  StreamingMessageType = 14
	TypeTensorSubscribe         StreamingMessageType = 15
	TypeTensorUnsubscribe       StreamingMessageType = 16
	TypeTensorNotify            StreamingMessageType = 17
	TypeTensorStreamAck         StreamingMessageType = 18
	TypeTensorStreamResume      StreamingMessageType = 19
	TypeTensorStreamCancel      StreamingMessageType = 20
	TypeTensorStreamInfo        StreamingMessageType = 21
	TypeTensorStreamStatus      StreamingMessageType = 22
	TypeTensorStreamMetadata    StreamingMessageType = 23
	TypeTensorHeartbeat         StreamingMessageType = 24
	
	// Compatibility aliases for the legacy types
	TypeTensorSync StreamingMessageType = 3 // Same as TypeTensorError
	TypeTensorAck  StreamingMessageType = 4
	TypeError      StreamingMessageType = 7
)

// StreamingHeader is the base header for all streaming messages
type StreamingHeader struct {
	// Header contains basic message information
	Header struct {
		// Type is the message type
		Type StreamingMessageType

		// MessageID is a unique identifier for this message
		MessageID uint32

		// CorrelationID is used to correlate request/response messages
		CorrelationID uint32

		// Timestamp is the unix timestamp when the message was created
		Timestamp uint64

		// TensorID is an identifier for the tensor
		TensorID string

		// Size is the total size of the message payload in bytes
		Size uint64
	}

	// ChunkNumber is the sequence number of this chunk
	ChunkNumber uint32

	// TotalChunks is the total number of chunks in the transfer
	TotalChunks uint32

	// Priority indicates the urgency of the request
	Priority uint8

	// CompressedSize is the size of the data after compression
	CompressedSize uint64

	// Checksum is a SHA-256 hash of the chunk data
	Checksum [32]byte

	// CompressionType indicates the compression algorithm used
	CompressionType CompressionType

	// ModelID identifies the model this tensor belongs to
	ModelID string

	// PartitionID identifies the partition within the model
	PartitionID string

	// Flags contains bit flags for various options
	Flags uint32
}


// StreamingMessage contains a header and data for streaming messages
type StreamingMessage struct {
	// Header contains metadata for this message
	Header StreamingHeader

	// Data contains the actual tensor data
	Data []byte
}

// CompressionOption represents configuration for a compression algorithm
type CompressionOption struct {
	// Type is the compression algorithm
	Type CompressionType

	// Level controls the compression level
	Level int

	// Threshold is the minimum size for compression to be applied
	Threshold uint64

	// Name is a human-readable name for the compression type
	Name string
}

// RetryPolicy defines how transfer retries are handled
type RetryPolicy struct {
	// MaxRetries is the maximum number of retry attempts
	MaxRetries int

	// BaseDelay is the initial delay between retries
	BaseDelay time.Duration

	// MaxDelay is the maximum delay between retries
	MaxDelay time.Duration

	// BackoffFactor is the multiplier applied to the delay after each retry
	BackoffFactor float64

	// JitterFactor adds randomness to the delay to prevent thundering herd
	JitterFactor float64
}

// TransferState represents the state of a tensor transfer
type TransferState struct {
	// StreamID uniquely identifies this transfer
	StreamID string

	// ModelID identifies the model
	ModelID string

	// PartitionID identifies the partition
	PartitionID string

	// TensorID identifies the tensor
	TensorID string

	// StartTime is when the transfer was initiated
	StartTime time.Time

	// LastUpdated is when the transfer state was last updated
	LastUpdated time.Time

	// BytesTransferred is how many bytes have been transferred
	BytesTransferred uint64

	// TotalBytes is the total size of the tensor
	TotalBytes uint64

	// ChunksTransferred is how many chunks have been transferred
	ChunksTransferred uint32

	// TotalChunks is the total number of chunks
	TotalChunks uint32

	// LastChunkReceived is the last chunk that was successfully received
	LastChunkReceived uint32

	// Status indicates the current transfer status
	Status string

	// Priority is the priority level of this transfer
	Priority PriorityLevel

	// CompressionType indicates which compression algorithm is being used
	CompressionType CompressionType

	// Error contains any error that occurred during transfer
	Error error

	// Checkpoints contains chunk numbers that have been verified
	Checkpoints []uint32

	// Metadata contains additional information about the tensor
	Metadata map[string]string
}

// NewTransferState creates a new transfer state for tracking
func NewTransferState(streamID, modelID, partitionID, tensorID string, totalBytes uint64, totalChunks uint32, priority PriorityLevel) *TransferState {
	return &TransferState{
		StreamID:         streamID,
		ModelID:          modelID,
		PartitionID:      partitionID,
		TensorID:         tensorID,
		StartTime:        time.Now(),
		LastUpdated:      time.Now(),
		BytesTransferred: 0,
		TotalBytes:       totalBytes,
		ChunksTransferred: 0,
		TotalChunks:      totalChunks,
		LastChunkReceived: 0,
		Status:           "initializing",
		Priority:         priority,
		CompressionType:  CompressionNone,
		Checkpoints:      make([]uint32, 0),
		Metadata:         make(map[string]string),
	}
}

// UpdateProgress updates the transfer progress
func (ts *TransferState) UpdateProgress(chunkNumber uint32, bytesReceived uint64) {
	ts.LastUpdated = time.Now()
	ts.LastChunkReceived = chunkNumber
	ts.ChunksTransferred++
	ts.BytesTransferred += bytesReceived

	// If we've received all chunks, mark as complete
	if ts.ChunksTransferred >= ts.TotalChunks {
		ts.Status = "complete"
	} else {
		ts.Status = "in_progress"
	}

	// Add checkpoint for every 10% of chunks
	if ts.TotalChunks >= 10 && chunkNumber%(ts.TotalChunks/10) == 0 {
		ts.Checkpoints = append(ts.Checkpoints, chunkNumber)
	}
}

// GetProgress returns the transfer progress as a percentage
func (ts *TransferState) GetProgress() float64 {
	if ts.TotalBytes == 0 {
		return 0
	}
	return float64(ts.BytesTransferred) / float64(ts.TotalBytes) * 100
}

// These functions have been moved to streaming_protocol.go to avoid duplication

// EncodeTensorID constructs a tensor ID from model, partition and local tensor ID
func EncodeTensorID(modelID, partitionID, localTensorID string) string {
	return fmt.Sprintf("%s:%s:%s", modelID, partitionID, localTensorID)
}

// DecodeTensorID extracts model, partition, and local tensor ID from a tensor ID
func DecodeTensorID(tensorID string) (modelID, partitionID, localTensorID string) {
	// Simple implementation for demonstration - in production would use more robust parsing
	parts := make([]string, 3)
	copy(parts, strings.SplitN(tensorID, ":", 3))
	if len(parts) >= 3 {
		return parts[0], parts[1], parts[2]
	}
	if len(parts) >= 2 {
		return parts[0], parts[1], ""
	}
	if len(parts) >= 1 {
		return parts[0], "", ""
	}
	return "", "", ""
}