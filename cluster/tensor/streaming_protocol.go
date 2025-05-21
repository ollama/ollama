package tensor

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"log/slog"
)

// Connection defines the interface required for tensor communications
type Connection interface {
	io.ReadWriter
	Close() error
	SetReadDeadline(t time.Time) error
	SetWriteDeadline(t time.Time) error
}

// StreamingProtocol implements tensor streaming over a network connection
type StreamingProtocol struct {
	// conn is the network connection
	conn Connection
	
	// sendMu protects concurrent sends on the connection
	sendMu sync.Mutex
	
	// recvMu protects concurrent receives on the connection
	recvMu sync.Mutex
	
	// nextMessageID is used to generate unique message IDs
	nextMessageID uint32
	
	// idMu protects nextMessageID
	idMu sync.Mutex

	// chunkSize is the size of each chunk in bytes
	chunkSize int

	// transferCache stores ongoing transfers
	transferCache map[string]*TransferState

	// cacheMutex protects the transfer cache
	cacheMutex sync.RWMutex

	// compressionOptions stores available compression options
	compressionOptions map[CompressionType]CompressionOption

	// compressor handles data compression/decompression
	compressor *Compressor

	// retryPolicy defines how retries are handled
	retryPolicy RetryPolicy

	// defaultPriority is the default priority for requests
	defaultPriority PriorityLevel

	// cleanupInterval is how often to clean up stale transfers
	cleanupInterval time.Duration

	// stopCleanup signals the cleanup routine to stop
	stopCleanup chan struct{}
}

// NewStreamingProtocol creates a new streaming protocol instance
func NewStreamingProtocol(conn Connection) *StreamingProtocol {
	// Initialize compressor with default config
	compressor := NewCompressor(DefaultCompressionConfig())

	// Initialize default compression options
	compressionOptions := map[CompressionType]CompressionOption{
		CompressionNone: {
			Type:      CompressionNone,
			Level:     0,
			Threshold: 0,
			Name:      "None",
		},
		CompressionLZ4: {
			Type:      CompressionLZ4,
			Level:     6,
			Threshold: 4 * 1024, // 4KB
			Name:      "LZ4",
		},
		CompressionDeflate: {
			Type:      CompressionDeflate,
			Level:     6,
			Threshold: 4 * 1024, // 4KB
			Name:      "Deflate",
		},
	}

	// Create streaming protocol
	sp := &StreamingProtocol{
		conn:               conn,
		nextMessageID:      1,
		chunkSize:          1 * 1024 * 1024, // 1MB default
		transferCache:      make(map[string]*TransferState),
		compressionOptions: compressionOptions,
		compressor:         compressor,
		defaultPriority:    PriorityNormal,
		cleanupInterval:    5 * time.Minute,
		stopCleanup:        make(chan struct{}),
		retryPolicy: RetryPolicy{
			MaxRetries:    3,
			BaseDelay:     500 * time.Millisecond,
			MaxDelay:      10 * time.Second,
			BackoffFactor: 2.0,
			JitterFactor:  0.1,
		},
	}

	return sp
}

// SetChunkSize sets the chunk size for streaming transfers
func (sp *StreamingProtocol) SetChunkSize(chunkSize int) {
	sp.chunkSize = chunkSize
}

// SetDefaultPriority sets the default priority for requests
func (sp *StreamingProtocol) SetDefaultPriority(priority PriorityLevel) {
	sp.defaultPriority = priority
}

// SetRetryPolicy configures the retry policy
func (sp *StreamingProtocol) SetRetryPolicy(policy RetryPolicy) {
	sp.retryPolicy = policy
}

// SetCompressionEnabled enables or disables data compression
func (sp *StreamingProtocol) SetCompressionEnabled(enabled bool) {
	// Set compression options
	if enabled {
		// Set compression type to LZ4 by default
		sp.compressionOptions[CompressionLZ4] = CompressionOption{
			Type:      CompressionLZ4,
			Level:     6,
			Threshold: 4 * 1024, // 4KB
			Name:      "LZ4",
		}
	} else {
		// Force everything to use no compression
		sp.compressionOptions[CompressionLZ4] = CompressionOption{
			Type:      CompressionNone,
			Level:     0,
			Threshold: uint64(1<<63 - 1), // Very high threshold to effectively disable
			Name:      "None",
		}
	}
	
	slog.Info("Compression setting updated",
		"enabled", enabled)
}

// SetCompressionThreshold sets the minimum size for compression to be applied
func (sp *StreamingProtocol) SetCompressionThreshold(threshold uint64) {
	// Update threshold for all compression types
	for typ, opt := range sp.compressionOptions {
		if typ != CompressionNone {
			opt.Threshold = threshold
			sp.compressionOptions[typ] = opt
		}
	}
	
	slog.Info("Compression threshold updated",
		"threshold", threshold)
}

// StartCleanupRoutine starts the background cleanup routine
func (sp *StreamingProtocol) StartCleanupRoutine() {
	go sp.cleanupRoutine()
}

// cleanupRoutine periodically cleans up stale transfers
func (sp *StreamingProtocol) cleanupRoutine() {
	ticker := time.NewTicker(sp.cleanupInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			sp.cleanupStaleTransfers()
		case <-sp.stopCleanup:
			return
		}
	}
}

// cleanupStaleTransfers removes transfers that have been inactive for too long
func (sp *StreamingProtocol) cleanupStaleTransfers() {
	now := time.Now()
	staleThreshold := 30 * time.Minute

	sp.cacheMutex.Lock()
	defer sp.cacheMutex.Unlock()

	for id, state := range sp.transferCache {
		// Skip active transfers
		if state.Status == "in_progress" {
			continue
		}

		// Remove completed/failed transfers after a delay
		if (state.Status == "complete" || state.Status == "failed") &&
			now.Sub(state.LastUpdated) > staleThreshold {
			delete(sp.transferCache, id)
			slog.Info("Removed stale transfer from cache",
				"transfer_id", id,
				"status", state.Status,
				"inactive_duration", now.Sub(state.LastUpdated))
		}
	}
}

// SendStreamingMessage sends a message using the streaming protocol
func (sp *StreamingProtocol) SendStreamingMessage(header StreamingHeader, data []byte) error {
	// Encode the header
	headerBytes := EncodeHeader(&header)

	// Get remote address for logging
	remoteAddr := "unknown"
	localAddr := "unknown"
	if conn, ok := sp.conn.(net.Conn); ok {
		remoteAddr = conn.RemoteAddr().String()
		if conn.LocalAddr() != nil {
			localAddr = conn.LocalAddr().String()
		}
	}
	
	// Enhanced Windows-specific connection configuration
	if runtime.GOOS == "windows" {
		// Only proceed with TCP-specific configurations if we have a TCP connection
		tcpConn, ok := sp.conn.(*net.TCPConn)
		if ok {
			// Set more modest buffer size for Windows - the 4MB buffer might be too large for slow networks
			bufferSize := 256 * 1024 // Use 256KB instead of 512KB to be more accommodating to slower networks
			if err := tcpConn.SetWriteBuffer(bufferSize); err != nil {
				slog.Warn("Failed to set TCP write buffer on Windows",
					"error", err,
					"buffer_size", bufferSize,
					"remote_addr", remoteAddr)
			}
			
			readBufSize := 256 * 1024
			if err := tcpConn.SetReadBuffer(readBufSize); err != nil {
				slog.Warn("Failed to set TCP read buffer on Windows",
					"error", err,
					"buffer_size", readBufSize,
					"remote_addr", remoteAddr)
			}
			
			// Disable Nagle's algorithm for streaming performance
			if err := tcpConn.SetNoDelay(true); err != nil {
				slog.Warn("Failed to disable Nagle algorithm on Windows TCP connection",
					"error", err,
					"remote_addr", remoteAddr)
			}
			
			// Set keep-alive to detect stale connections
			if err := tcpConn.SetKeepAlive(true); err != nil {
				slog.Warn("Failed to enable keep-alive on Windows TCP connection",
					"error", err,
					"remote_addr", remoteAddr)
			} else {
				// Set a more aggressive keep-alive period for Windows
				if err := tcpConn.SetKeepAlivePeriod(30 * time.Second); err != nil {
					slog.Warn("Failed to set keep-alive period on Windows TCP connection",
						"error", err,
						"remote_addr", remoteAddr)
				}
			}

			// Add network binding diagnostics to help troubleshoot IP binding issues
			connLocalAddr := tcpConn.LocalAddr().String()
			localAddrType := getIPType(connLocalAddr)
			remoteAddrType := getIPType(remoteAddr)

			// Log warning for potential network binding issues that can affect cluster communication
			if localAddrType == "loopback" && remoteAddrType != "loopback" && remoteAddrType != "unknown" {
				slog.Warn("Possible Windows network binding issue detected: using loopback for non-local connection",
					"local_addr", connLocalAddr,
					"local_type", localAddrType,
					"remote_addr", remoteAddr,
					"remote_type", remoteAddrType,
					"recommendation", "Check firewall settings and bind specific IP addresses")
			}
			
			// Log detailed network information to help diagnose IP address issues
			slog.Info("Applied Windows-specific TCP optimizations",
				"read_buffer", readBufSize,
				"write_buffer", bufferSize,
				"no_delay", true,
				"keep_alive", true,
				"remote_addr", remoteAddr,
				"local_addr", localAddr,
				"remote_ip_type", getIPType(remoteAddr),
				"local_ip_type", getIPType(localAddr))
		}
	}

	// Log detailed operation info
	var msgDesc string
	switch header.Header.Type {
	case TypeTensorStreamInfo:
		msgDesc = "stream initialization"
	case TypeTensorStreamChunk:
		msgDesc = fmt.Sprintf("chunk %d/%d", header.ChunkNumber, header.TotalChunks)
	case TypeTensorStreamComplete:
		msgDesc = "stream completion"
	case TypeTensorStreamRequest:
		msgDesc = "tensor request"
	default:
		msgDesc = fmt.Sprintf("type %d", header.Header.Type)
	}

	slog.Info("Sending streaming message",
		"message_type", header.Header.Type,
		"description", msgDesc,
		"tensor_id", header.Header.TensorID,
		"model_id", header.ModelID,
		"data_size", len(data),
		"remote_addr", remoteAddr)

	sp.sendMu.Lock()
	defer sp.sendMu.Unlock()

	// Setup write deadline for this operation
	if tcpConn, ok := sp.conn.(*net.TCPConn); ok {
		deadline := time.Now().Add(30 * time.Second) // 30-second timeout
		if err := tcpConn.SetWriteDeadline(deadline); err != nil {
			slog.Warn("Failed to set write deadline", "error", err)
			// Continue despite failure to set deadline
		}

		// Check for loopback connections when we should be using network IPs
		if runtime.GOOS == "windows" && remoteAddr != "unknown" {
			localAddr := "unknown"
			if tcpConn.LocalAddr() != nil {
				localAddr = tcpConn.LocalAddr().String()
			}
			
			remoteIP := getIPType(remoteAddr)
			localIP := getIPType(localAddr)
			
			// Log warning if we're using loopback for non-loopback destinations
			if localIP == "loopback" && remoteIP != "loopback" && remoteIP != "unknown" {
				slog.Warn("Windows network issue detected: using loopback interface for non-loopback destination",
					"remote_addr", remoteAddr,
					"remote_ip_type", remoteIP,
					"local_addr", localAddr,
					"local_ip_type", localIP,
					"recommendation", "Check firewall settings and network binding configuration")
			}
		}
	}

	// Write the header size (uint32)
	headerSizeBytes := make([]byte, 4)
	binary.LittleEndian.PutUint32(headerSizeBytes, uint32(len(headerBytes)))
	if _, err := sp.conn.Write(headerSizeBytes); err != nil {
		// Enhanced Windows-specific error diagnostics
		var netErr net.Error
		timeout := errors.As(err, &netErr) && netErr.Timeout()
		isEOF := err == io.EOF
		errStr := err.Error()
		isWindowsNetworkErr := strings.Contains(errStr, "wsarecv") ||
			strings.Contains(errStr, "wsasend") ||
			strings.Contains(errStr, "WSAECONNRESET")
		forcedClose := strings.Contains(errStr, "forcibly closed") ||
			strings.Contains(errStr, "connection reset by peer") ||
			strings.Contains(errStr, "broken pipe")
		
		slog.Error("Failed to write header size",
			"error", err,
			"error_type", fmt.Sprintf("%T", err),
			"error_message", errStr,
			"is_timeout", timeout,
			"is_eof", isEOF,
			"is_windows_error", isWindowsNetworkErr && runtime.GOOS == "windows",
			"forced_close", forcedClose,
			"message_type", header.Header.Type,
			"remote_addr", remoteAddr,
			"local_addr", localAddr,
			"os", runtime.GOOS,
			"data_size", len(data))
		
		return fmt.Errorf("failed to write header size: %w", err)
	}

	// Write the header
	if _, err := sp.conn.Write(headerBytes); err != nil {
		slog.Error("Failed to write header",
			"error", err,
			"error_type", fmt.Sprintf("%T", err),
			"header_size", len(headerBytes),
			"message_type", header.Header.Type,
			"remote_addr", remoteAddr)
		return fmt.Errorf("failed to write header: %w", err)
	}

	// Write the data
	if len(data) > 0 {
		// For larger chunks, log progress
		if len(data) > 1024*1024 { // 1MB
			slog.Info("Starting large data write",
				"size", len(data),
				"message_type", header.Header.Type,
				"tensor_id", header.Header.TensorID)
		}
		
		if _, err := sp.conn.Write(data); err != nil {
			slog.Error("Failed to write message data",
				"error", err,
				"error_type", fmt.Sprintf("%T", err),
				"data_size", len(data),
				"message_type", header.Header.Type,
				"remote_addr", remoteAddr)
			return fmt.Errorf("failed to write data: %w", err)
		}
	}

	return nil
}

// ReceiveStreamingMessage receives a message using the streaming protocol
func (sp *StreamingProtocol) ReceiveStreamingMessage() (*StreamingHeader, []byte, error) {
	sp.recvMu.Lock()
	defer sp.recvMu.Unlock()

	// Add detailed connection logging
	remoteAddr := "unknown"
	localAddr := "unknown"
	if conn, ok := sp.conn.(net.Conn); ok {
		remoteAddr = conn.RemoteAddr().String()
		if conn.LocalAddr() != nil {
			localAddr = conn.LocalAddr().String()
		}
	}
	
	// Set connection timeout - if we don't receive anything within 30 seconds, fail rather than hang
	if tcpConn, ok := sp.conn.(*net.TCPConn); ok {
		readDeadline := time.Now().Add(30 * time.Second)
		if err := tcpConn.SetReadDeadline(readDeadline); err != nil {
			slog.Warn("Failed to set read deadline",
				"error", err,
				"remote_addr", remoteAddr)
			// Continue anyway, but log the warning
		}
	}

	slog.Info("Waiting for streaming message",
		"remote_addr", remoteAddr,
		"local_addr", localAddr)

	// Read the header size (uint32)
	headerSizeBytes := make([]byte, 4)
	startTime := time.Now()
	
	// Windows-specific retry mechanism for transient network issues
	maxRetries := 3
	retryDelay := 50 * time.Millisecond
	var readHeaderSizeErr error
	
	for retry := 0; retry <= maxRetries; retry++ {
		err := func() error {
			_, err := io.ReadFull(sp.conn, headerSizeBytes)
			return err
		}()
		
		if err == nil {
			readHeaderSizeErr = nil
			break
		}
		
		if retry < maxRetries && runtime.GOOS == "windows" {
			// Only retry on Windows for specific network errors
			isNetErr, errType := IsNetworkError(err)
			if isNetErr && strings.HasPrefix(errType, "windows_socket_error_") {
				slog.Warn("Retrying Windows network read after error",
					"error", err,
					"retry", retry+1,
					"max_retries", maxRetries,
					"remote_addr", remoteAddr)
					
				time.Sleep(retryDelay)
				retryDelay *= 2 // Exponential backoff
				continue
			}
		}
		
		readHeaderSizeErr = err
		break
	}
	
	if readHeaderSizeErr != nil {
		// Enhanced error classification and logging
		var netErr net.Error
		timeout := errors.As(readHeaderSizeErr, &netErr) && netErr.Timeout()
		isEOF := readHeaderSizeErr == io.EOF
		forcedClose := strings.Contains(readHeaderSizeErr.Error(), "forcibly closed") ||
			strings.Contains(readHeaderSizeErr.Error(), "connection reset by peer") ||
			strings.Contains(readHeaderSizeErr.Error(), "broken pipe") ||
			strings.Contains(readHeaderSizeErr.Error(), "wsarecv")
		
		slog.Error("Failed to read header size",
			"error", readHeaderSizeErr,
			"error_type", fmt.Sprintf("%T", readHeaderSizeErr),
			"error_msg", readHeaderSizeErr.Error(),
			"is_timeout", timeout,
			"is_eof", isEOF,
			"forced_close", forcedClose,
			"elapsed_ms", time.Since(startTime).Milliseconds(),
			"remote_addr", remoteAddr,
			"local_addr", localAddr)
		
		if forcedClose {
			slog.Error("Connection forcibly closed during header size read - network issue detected",
				"remote_addr", remoteAddr,
				"local_addr", localAddr)
		}
		
		return nil, nil, fmt.Errorf("failed to read header size: %w", readHeaderSizeErr)
	}
	
	// Successfully read header size
	headerSize := binary.LittleEndian.Uint32(headerSizeBytes)
	
	// Validate header size is reasonable
	if headerSize < 10 || headerSize > 1024*1024 { // Min 10 bytes, max 1MB for a header
		slog.Error("Invalid header size received",
			"size", headerSize,
			"remote_addr", remoteAddr)
		return nil, nil, fmt.Errorf("invalid header size: %d", headerSize)
	}
	
	slog.Debug("Header size received",
		"size", headerSize,
		"remote_addr", remoteAddr)

	// Read the header
	headerBytes := make([]byte, headerSize)
	if _, err := io.ReadFull(sp.conn, headerBytes); err != nil {
		// Check if this is a network error
		isNetwork, errType := IsNetworkError(err)
		
		slog.Error("Failed to read header content",
			"error", err,
			"error_type", fmt.Sprintf("%T", err),
			"is_network_error", isNetwork,
			"network_error_type", errType,
			"header_size", headerSize,
			"remote_addr", remoteAddr,
			"local_addr", localAddr)
		return nil, nil, fmt.Errorf("failed to read header: %w", err)
	}

	// Decode the header
	header, err := DecodeHeader(headerBytes)
	if err != nil {
		slog.Error("Failed to decode header",
			"error", err,
			"error_type", fmt.Sprintf("%T", err),
			"header_size", headerSize,
			"header_bytes", fmt.Sprintf("%x", headerBytes[:min(len(headerBytes), 32)]), // First 32 bytes as hex
			"remote_addr", remoteAddr)
		return nil, nil, fmt.Errorf("failed to decode header: %w", err)
	}

	// Create message type name for more helpful logging
	var msgTypeName string
	switch header.Header.Type {
	case TypeTensorStreamInfo:
		msgTypeName = "TensorStreamInfo"
	case TypeTensorStreamChunk:
		msgTypeName = "TensorStreamChunk"
	case TypeTensorStreamComplete:
		msgTypeName = "TensorStreamComplete"
	case TypeTensorStreamRequest:
		msgTypeName = "TensorStreamRequest"
	default:
		msgTypeName = fmt.Sprintf("Unknown(%d)", header.Header.Type)
	}

	// Read the data
	var data []byte
	if header.Header.Size > 0 {
		// Validate data size is reasonable
		if header.Header.Size > 100*1024*1024 {
			slog.Error("Excessively large message size",
				"size", header.Header.Size,
				"type", msgTypeName,
				"remote_addr", remoteAddr)
			return nil, nil, fmt.Errorf("message size too large: %d bytes", header.Header.Size)
		}
		
		slog.Debug("Reading message data",
			"size", header.Header.Size,
			"type", msgTypeName,
			"tensor_id", header.Header.TensorID,
			"remote_addr", remoteAddr)
		
		data = make([]byte, header.Header.Size)
		readStartTime := time.Now()
		if _, err := io.ReadFull(sp.conn, data); err != nil {
			// Check if this is a network error
			isNetwork, errType := IsNetworkError(err)
			
			slog.Error("Failed to read message data",
				"error", err,
				"error_type", fmt.Sprintf("%T", err),
				"error_msg", err.Error(),
				"is_network_error", isNetwork,
				"network_error_type", errType,
				"expected_size", header.Header.Size,
				"message_type", msgTypeName,
				"elapsed_ms", time.Since(readStartTime).Milliseconds(),
				"remote_addr", remoteAddr)
				
			if isNetwork && errType == "connection_closed" {
				slog.Error("Connection closed during data transfer - this may indicate a system resource issue",
					"message_size", header.Header.Size,
					"message_type", msgTypeName)
			}
			
			return nil, nil, fmt.Errorf("failed to read data: %w", err)
		}
	}

	slog.Debug("Successfully received streaming message",
		"type", msgTypeName,
		"size", header.Header.Size,
		"tensor_id", header.Header.TensorID,
		"remote_addr", remoteAddr)

	return header, data, nil
}

// StreamTensor streams a tensor in chunks
func (sp *StreamingProtocol) StreamTensor(modelID, partitionID, tensorID string, tensorData []byte) error {
	// Generate a unique stream ID
	streamID := GenerateStreamID(modelID, partitionID, tensorID)

	// Calculate the total number of chunks
	totalChunks := (len(tensorData) + sp.chunkSize - 1) / sp.chunkSize

	// Initialize transfer state
	transferState := NewTransferState(
		streamID,
		modelID,
		partitionID,
		tensorID,
		uint64(len(tensorData)),
		uint32(totalChunks),
		sp.defaultPriority,
	)

	// Store in cache
	sp.cacheMutex.Lock()
	sp.transferCache[streamID] = transferState
	sp.cacheMutex.Unlock()

	// Send initialization message
	initHeader := StreamingHeader{
		Header: struct {
			Type StreamingMessageType
			MessageID uint32
			CorrelationID uint32
			Timestamp uint64
			TensorID string
			Size uint64
		}{
			Type:          TypeTensorStreamInfo,
			MessageID:     uint32(time.Now().UnixNano()),
			CorrelationID: 0,
			Timestamp:     uint64(time.Now().Unix()),
			TensorID:      tensorID,
			Size:          0, // No data in this message
		},
		ChunkNumber:     0,
		TotalChunks:     uint32(totalChunks),
		Priority:        uint8(sp.defaultPriority),
		CompressedSize:  0,
		Checksum:        [32]byte{},
		CompressionType: CompressionNone,
		ModelID:         modelID,
		PartitionID:     partitionID,
		Flags:           0,
	}

	if err := sp.SendStreamingMessage(initHeader, nil); err != nil {
		return fmt.Errorf("failed to send initialization message: %w", err)
	}

	// Send each chunk
	for chunkIndex := 0; chunkIndex < totalChunks; chunkIndex++ {
		// Calculate chunk bounds
		start := chunkIndex * sp.chunkSize
		end := start + sp.chunkSize
		if end > len(tensorData) {
			end = len(tensorData)
		}

		// Extract chunk data
		chunkData := tensorData[start:end]

		// Compress the chunk if beneficial
		var compressedData []byte
		var compressionType CompressionType
		var compressedSize uint64

		compressedData, stats, err := sp.compressor.CompressWithStats(chunkData, CompressionLZ4)
		if err != nil {
			slog.Warn("Compression failed, sending uncompressed",
				"error", err,
				"chunk", chunkIndex)
			compressedData = chunkData
			compressionType = CompressionNone
			compressedSize = uint64(len(chunkData))
		} else {
			isBeneficial, _ := stats["beneficial"].(bool)
			if isBeneficial {
				compressionType = CompressionLZ4
				compressedSize = uint64(len(compressedData))
			} else {
				compressedData = chunkData
				compressionType = CompressionNone
				compressedSize = uint64(len(chunkData))
			}
		}

		// Calculate checksum for verification
		checksum := CalculateChecksum(chunkData)

		// Create header for this chunk
		chunkHeader := StreamingHeader{
			Header: struct {
				Type StreamingMessageType
				MessageID uint32
				CorrelationID uint32
				Timestamp uint64
				TensorID string
				Size uint64
			}{
				Type:          TypeTensorStreamChunk,
				MessageID:     uint32(time.Now().UnixNano()),
				CorrelationID: 0,
				Timestamp:     uint64(time.Now().Unix()),
				TensorID:      tensorID,
				Size:          uint64(len(compressedData)),
			},
			ChunkNumber:     uint32(chunkIndex),
			TotalChunks:     uint32(totalChunks),
			Priority:        uint8(sp.defaultPriority),
			CompressedSize:  compressedSize,
			Checksum:        checksum,
			CompressionType: compressionType,
			ModelID:         modelID,
			PartitionID:     partitionID,
			Flags:           0,
		}

		// Send the chunk with retry logic
		err = sp.sendChunkWithRetry(chunkHeader, compressedData)
		if err != nil {
			// Update transfer state to failed
			sp.cacheMutex.Lock()
			if ts, exists := sp.transferCache[streamID]; exists {
				ts.Status = "failed"
				ts.Error = err
				ts.LastUpdated = time.Now()
			}
			sp.cacheMutex.Unlock()
			return fmt.Errorf("failed to send chunk %d: %w", chunkIndex, err)
		}

		// Update transfer state
		sp.cacheMutex.Lock()
		if ts, exists := sp.transferCache[streamID]; exists {
			ts.UpdateProgress(uint32(chunkIndex), uint64(len(chunkData)))
		}
		sp.cacheMutex.Unlock()
	}

	// Send completion message
	completeHeader := StreamingHeader{
		Header: struct {
			Type StreamingMessageType
			MessageID uint32
			CorrelationID uint32
			Timestamp uint64
			TensorID string
			Size uint64
		}{
			Type:          TypeTensorStreamComplete,
			MessageID:     uint32(time.Now().UnixNano()),
			CorrelationID: 0,
			Timestamp:     uint64(time.Now().Unix()),
			TensorID:      tensorID,
			Size:          0, // No data in this message
		},
		ChunkNumber:     uint32(totalChunks),
		TotalChunks:     uint32(totalChunks),
		Priority:        uint8(sp.defaultPriority),
		CompressedSize:  0,
		Checksum:        [32]byte{},
		CompressionType: CompressionNone,
		ModelID:         modelID,
		PartitionID:     partitionID,
		Flags:           0,
	}

	if err := sp.SendStreamingMessage(completeHeader, nil); err != nil {
		return fmt.Errorf("failed to send completion message: %w", err)
	}

	// Update transfer state to complete
	sp.cacheMutex.Lock()
	if ts, exists := sp.transferCache[streamID]; exists {
		ts.Status = "complete"
		ts.LastUpdated = time.Now()
	}
	sp.cacheMutex.Unlock()

	slog.Info("Successfully streamed tensor",
		"model_id", modelID,
		"partition_id", partitionID,
		"tensor_id", tensorID,
		"stream_id", streamID,
		"chunks", totalChunks,
		"total_bytes", len(tensorData))

	return nil
}

// sendChunkWithRetry sends a chunk with retry logic
func (sp *StreamingProtocol) sendChunkWithRetry(header StreamingHeader, data []byte) error {
	var err error
	delay := sp.retryPolicy.BaseDelay

	for retry := 0; retry <= sp.retryPolicy.MaxRetries; retry++ {
		// Attempt to send
		err = sp.SendStreamingMessage(header, data)
		if err == nil {
			return nil // Success
		}

		// If this was the last retry, return the error
		if retry == sp.retryPolicy.MaxRetries {
			return fmt.Errorf("failed after %d retries: %w", retry, err)
		}

		// Log the retry attempt
		slog.Warn("Failed to send chunk, retrying",
			"chunk", header.ChunkNumber,
			"retry", retry+1,
			"max_retries", sp.retryPolicy.MaxRetries,
			"delay", delay,
			"error", err)

		// Wait before retrying
		time.Sleep(delay)

		// Increase delay for next retry (exponential backoff)
		delay = time.Duration(float64(delay) * sp.retryPolicy.BackoffFactor)
		if delay > sp.retryPolicy.MaxDelay {
			delay = sp.retryPolicy.MaxDelay
		}
	}

	return err // Should never reach here due to the return in the loop
}

// RequestTensor requests a tensor from a remote node with priority
func (sp *StreamingProtocol) RequestTensor(modelID, partitionID, tensorID string, priority PriorityLevel) (string, error) {
	// Generate a unique stream ID
	streamID := GenerateStreamID(modelID, partitionID, tensorID)

	// Initialize transfer state
	transferState := NewTransferState(
		streamID,
		modelID,
		partitionID,
		tensorID,
		0, // Unknown size yet
		0, // Unknown chunks yet
		priority,
	)

	// Store in cache
	sp.cacheMutex.Lock()
	sp.transferCache[streamID] = transferState
	sp.cacheMutex.Unlock()

	// Create request header
	requestHeader := StreamingHeader{
		Header: struct {
			Type StreamingMessageType
			MessageID uint32
			CorrelationID uint32
			Timestamp uint64
			TensorID string
			Size uint64
		}{
			Type:          TypeTensorStreamRequest,
			MessageID:     uint32(time.Now().UnixNano()),
			CorrelationID: 0,
			Timestamp:     uint64(time.Now().Unix()),
			TensorID:      tensorID,
			Size:          0, // No data in this message
		},
		ChunkNumber:     0,
		TotalChunks:     0,
		Priority:        uint8(priority),
		CompressedSize:  0,
		Checksum:        [32]byte{},
		CompressionType: CompressionNone,
		ModelID:         modelID,
		PartitionID:     partitionID,
		Flags:           0,
	}

	// Send the request
	if err := sp.SendStreamingMessage(requestHeader, nil); err != nil {
		return "", fmt.Errorf("failed to send tensor request: %w", err)
	}

	slog.Info("Requested tensor",
		"model_id", modelID,
		"partition_id", partitionID,
		"tensor_id", tensorID,
		"stream_id", streamID,
		"priority", priority)

	return streamID, nil
}

// ResumeTensorTransfer resumes an interrupted transfer
func (sp *StreamingProtocol) ResumeTensorTransfer(streamID string) (bool, error) {
	// Get transfer state
	sp.cacheMutex.RLock()
	ts, exists := sp.transferCache[streamID]
	sp.cacheMutex.RUnlock()

	if !exists {
		return false, fmt.Errorf("transfer %s not found", streamID)
	}

	// Cannot resume completed or non-interrupted transfers
	if ts.Status == "complete" {
		return false, nil
	}

	// Create resume request header
	resumeHeader := StreamingHeader{
		Header: struct {
			Type StreamingMessageType
			MessageID uint32
			CorrelationID uint32
			Timestamp uint64
			TensorID string
			Size uint64
		}{
			Type:          TypeTensorStreamResume,
			MessageID:     uint32(time.Now().UnixNano()),
			CorrelationID: 0,
			Timestamp:     uint64(time.Now().Unix()),
			TensorID:      ts.TensorID,
			Size:          0, // No data in this message
		},
		ChunkNumber:     ts.LastChunkReceived + 1, // Start from next chunk
		TotalChunks:     ts.TotalChunks,
		Priority:        uint8(ts.Priority),
		CompressedSize:  0,
		Checksum:        [32]byte{},
		CompressionType: CompressionNone,
		ModelID:         ts.ModelID,
		PartitionID:     ts.PartitionID,
		Flags:           0,
	}

	// Send the resume request
	if err := sp.SendStreamingMessage(resumeHeader, nil); err != nil {
		return false, fmt.Errorf("failed to send resume request: %w", err)
	}

	// Update transfer state
	sp.cacheMutex.Lock()
	if ts, exists := sp.transferCache[streamID]; exists {
		ts.Status = "in_progress"
		ts.LastUpdated = time.Now()
	}
	sp.cacheMutex.Unlock()

	slog.Info("Resumed tensor transfer",
		"stream_id", streamID,
		"tensor_id", ts.TensorID,
		"last_chunk", ts.LastChunkReceived,
		"total_chunks", ts.TotalChunks)

	return true, nil
}

// PrefetchTensors requests tensors likely to be needed soon
func (sp *StreamingProtocol) PrefetchTensors(modelID, partitionID string, tensorIDs []string, priority PriorityLevel) ([]string, error) {
	streamIDs := make([]string, 0, len(tensorIDs))

	// Create prefetch request header
	for _, tensorID := range tensorIDs {
		// Generate a unique stream ID
		streamID := GenerateStreamID(modelID, partitionID, tensorID)
		streamIDs = append(streamIDs, streamID)

		prefetchHeader := StreamingHeader{
			Header: struct {
				Type StreamingMessageType
				MessageID uint32
				CorrelationID uint32
				Timestamp uint64
				TensorID string
				Size uint64
			}{
				Type:          TypeTensorPrefetchRequest,
				MessageID:     uint32(time.Now().UnixNano()),
				CorrelationID: 0,
				Timestamp:     uint64(time.Now().Unix()),
				TensorID:      tensorID,
				Size:          0, // No data in this message
			},
			ChunkNumber:     0,
			TotalChunks:     0,
			Priority:        uint8(priority),
			CompressedSize:  0,
			Checksum:        [32]byte{},
			CompressionType: CompressionNone,
			ModelID:         modelID,
			PartitionID:     partitionID,
			Flags:           0,
		}

		// Send the prefetch request
		if err := sp.SendStreamingMessage(prefetchHeader, nil); err != nil {
			return streamIDs, fmt.Errorf("failed to send prefetch request for %s: %w", tensorID, err)
		}

		// Initialize transfer state
		transferState := NewTransferState(
			streamID,
			modelID,
			partitionID,
			tensorID,
			0, // Unknown size yet
			0, // Unknown chunks yet
			priority,
		)

		// Store in cache
		sp.cacheMutex.Lock()
		sp.transferCache[streamID] = transferState
		sp.cacheMutex.Unlock()
	}

	slog.Info("Prefetched tensors",
		"model_id", modelID,
		"partition_id", partitionID,
		"tensor_count", len(tensorIDs),
		"priority", priority)

	return streamIDs, nil
}

// IsNetworkError checks if an error is a network-related error with enhanced Windows support
func IsNetworkError(err error) (bool, string) {
	if err == nil {
		return false, ""
	}
	
	errStr := err.Error()
	
	// Windows-specific errors with enhanced diagnostics
	if runtime.GOOS == "windows" {
		if strings.Contains(errStr, "wsarecv") || strings.Contains(errStr, "wsasend") {
			errorCode := "unknown"
			
			// Try to extract the Windows error code if present
			if strings.Contains(errStr, "WSAECONNRESET") {
				errorCode = "WSAECONNRESET"
			} else if strings.Contains(errStr, "WSAETIMEDOUT") {
				errorCode = "WSAETIMEDOUT"
			} else if strings.Contains(errStr, "WSAECONNABORTED") {
				errorCode = "WSAECONNABORTED"
			}
			
			slog.Error("Windows socket error detected",
				"error", err,
				"error_code", errorCode,
				"os", runtime.GOOS,
				"error_type", "windows_socket",
				"details", "Socket operation failure, possible network interruption")
			return true, "windows_socket_error_" + errorCode
		}
	}
	
	// Other connection closed errors
	if strings.Contains(errStr, "forcibly closed") ||
	   strings.Contains(errStr, "connection reset by peer") ||
	   strings.Contains(errStr, "broken pipe") {
		slog.Error("Connection forcibly closed",
			"error", err,
			"os", runtime.GOOS)
		return true, "connection_closed"
	}
	
	// Timeout errors
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true, "timeout"
	}
	
	// EOF errors
	if err == io.EOF {
		return true, "eof"
	}
	
	return false, ""
}

// HandleStreamingMessage handles incoming streaming messages
func (sp *StreamingProtocol) HandleStreamingMessage(header *StreamingHeader, data []byte) error {
	// Get remote address for logging
	remoteAddr := "unknown"
	localAddr := "unknown"
	if conn, ok := sp.conn.(net.Conn); ok {
		remoteAddr = conn.RemoteAddr().String()
		if conn.LocalAddr() != nil {
			localAddr = conn.LocalAddr().String()
		}
	}
	
	// Set a reasonable timeout for handlers
	if tcpConn, ok := sp.conn.(*net.TCPConn); ok {
		deadline := time.Now().Add(60 * time.Second)
		if err := tcpConn.SetReadDeadline(deadline); err != nil {
			slog.Warn("Failed to set read deadline", "error", err)
		}
	}
	
	slog.Info("Handling streaming message",
		"message_type", header.Header.Type,
		"tensor_id", header.Header.TensorID,
		"data_size", len(data),
		"model_id", header.ModelID,
		"remote_addr", remoteAddr)
	
	// Create more descriptive message type name
	var msgTypeName string
	switch header.Header.Type {
	case TypeTensorStreamInfo:
		msgTypeName = "TensorStreamInfo"
	case TypeTensorStreamChunk:
		msgTypeName = "TensorStreamChunk"
	case TypeTensorStreamComplete:
		msgTypeName = "TensorStreamComplete"
	case TypeTensorStreamRequest:
		msgTypeName = "TensorStreamRequest"
	case TypeTensorStreamResume:
		msgTypeName = "TensorStreamResume"
	case TypeTensorPrefetchRequest:
		msgTypeName = "TensorPrefetchRequest"
	case TypeTensorStreamCancel:
		msgTypeName = "TensorStreamCancel"
	case TypeTensorHeartbeat:
		msgTypeName = "TensorHeartbeat"
	default:
		msgTypeName = fmt.Sprintf("Unknown(%d)", header.Header.Type)
	}
	
	// Track performance for debugging
	startTime := time.Now()
	
	// Generate stream ID for logging/tracking
	streamID := ""
	if header.ModelID != "" && header.PartitionID != "" && header.Header.TensorID != "" {
		streamID = GenerateStreamID(header.ModelID, header.PartitionID, header.Header.TensorID)
		slog.Info("Processing tensor operation",
			"stream_id", streamID,
			"message_type", msgTypeName)
	}
	
	var err error
	switch header.Header.Type {
	case TypeTensorStreamInfo:
		err = sp.handleTensorStreamInfo(header, data)
	case TypeTensorStreamChunk:
		err = sp.handleTensorStreamChunk(header, data)
	case TypeTensorStreamComplete:
		err = sp.handleTensorStreamComplete(header, data)
	case TypeTensorStreamRequest:
		err = sp.handleTensorStreamRequest(header, data)
	case TypeTensorStreamResume:
		err = sp.handleTensorStreamResume(header, data)
	case TypeTensorPrefetchRequest:
		err = sp.handleTensorPrefetchRequest(header, data)
	case TypeTensorStreamCancel:
		err = sp.handleTensorStreamCancel(header, data)
	case TypeTensorHeartbeat:
		err = sp.handleTensorHeartbeat(header, data)
	default:
		err = fmt.Errorf("unknown message type: %d", header.Header.Type)
	}
	
	// Log performance and result
	duration := time.Since(startTime)
	if err != nil {
		slog.Error("Error handling streaming message",
			"message_type", msgTypeName,
			"error", err,
			"error_type", fmt.Sprintf("%T", err),
			"duration_ms", duration.Milliseconds(),
			"stream_id", streamID,
			"remote_addr", remoteAddr)
		
		// Check if this is a network-related error
		isNetwork, errType := IsNetworkError(err)
		if isNetwork {
			slog.Error("Network error detected in handler",
				"error_type", errType,
				"message_type", msgTypeName,
				"stream_id", streamID,
				"remote_addr", remoteAddr,
				"local_addr", localAddr)
		}
	} else {
		slog.Debug("Successfully handled streaming message",
			"message_type", msgTypeName,
			"duration_ms", duration.Milliseconds(),
			"stream_id", streamID,
			"remote_addr", remoteAddr)
	}
	
	return err
}

// handleTensorStreamInfo handles tensor stream info messages
func (sp *StreamingProtocol) handleTensorStreamInfo(header *StreamingHeader, data []byte) error {
	// Extract info from header
	modelID := header.ModelID
	partitionID := header.PartitionID
	tensorID := header.Header.TensorID
	totalChunks := header.TotalChunks

	// Generate stream ID
	streamID := GenerateStreamID(modelID, partitionID, tensorID)

	// Create or update transfer state
	sp.cacheMutex.Lock()
	defer sp.cacheMutex.Unlock()

	ts, exists := sp.transferCache[streamID]
	if !exists {
		// Create new transfer state
		ts = NewTransferState(
			streamID,
			modelID,
			partitionID,
			tensorID,
			0, // Unknown size yet
			totalChunks,
			PriorityLevel(header.Priority),
		)
		sp.transferCache[streamID] = ts
	} else {
		// Update existing transfer state
		ts.TotalChunks = totalChunks
		ts.Priority = PriorityLevel(header.Priority)
		ts.Status = "initializing"
		ts.LastUpdated = time.Now()
	}

	slog.Info("Received tensor stream info",
		"model_id", modelID,
		"partition_id", partitionID,
		"tensor_id", tensorID,
		"stream_id", streamID,
		"total_chunks", totalChunks)

	return nil
}

// handleTensorStreamChunk handles tensor stream chunk messages
func (sp *StreamingProtocol) handleTensorStreamChunk(header *StreamingHeader, data []byte) error {
	// Extract info from header
	modelID := header.ModelID
	partitionID := header.PartitionID
	tensorID := header.Header.TensorID
	chunkNumber := header.ChunkNumber
	originalSize := header.CompressedSize
	compressionType := header.CompressionType
	checksum := header.Checksum

	// Generate stream ID
	streamID := GenerateStreamID(modelID, partitionID, tensorID)

	// Get transfer state
	sp.cacheMutex.Lock()
	ts, exists := sp.transferCache[streamID]
	if !exists {
		// Create new transfer state
		ts = NewTransferState(
			streamID,
			modelID,
			partitionID,
			tensorID,
			0, // Unknown total size
			0, // Unknown total chunks
			PriorityLevel(header.Priority),
		)
		sp.transferCache[streamID] = ts
	}
	sp.cacheMutex.Unlock()

	// Decompress the data if needed
	var decompressedData []byte
	var err error
	if compressionType != CompressionNone {
		decompressedData, err = sp.compressor.Decompress(data, compressionType, originalSize)
		if err != nil {
			return fmt.Errorf("failed to decompress chunk: %w", err)
		}
	} else {
		decompressedData = data
	}

	// Verify checksum
	calculatedChecksum := CalculateChecksum(decompressedData)
	if calculatedChecksum != checksum {
		return fmt.Errorf("checksum mismatch for chunk %d", chunkNumber)
	}

	// Process the chunk data (in a real implementation, would store in memory or disk)
	// For this example, we'll just update the transfer state
	sp.cacheMutex.Lock()
	if ts, exists := sp.transferCache[streamID]; exists {
		ts.UpdateProgress(chunkNumber, uint64(len(decompressedData)))
	}
	sp.cacheMutex.Unlock()

	// Send acknowledgment
	ackHeader := StreamingHeader{
		Header: struct {
			Type StreamingMessageType
			MessageID uint32
			CorrelationID uint32
			Timestamp uint64
			TensorID string
			Size uint64
		}{
			Type:          TypeTensorStreamAck,
			MessageID:     uint32(time.Now().UnixNano()),
			CorrelationID: header.Header.MessageID,
			Timestamp:     uint64(time.Now().Unix()),
			TensorID:      tensorID,
			Size:          0, // No data in this message
		},
		ChunkNumber:     chunkNumber,
		TotalChunks:     header.TotalChunks,
		Priority:        header.Priority,
		CompressedSize:  0,
		Checksum:        [32]byte{},
		CompressionType: CompressionNone,
		ModelID:         modelID,
		PartitionID:     partitionID,
		Flags:           0,
	}

	if err := sp.SendStreamingMessage(ackHeader, nil); err != nil {
		slog.Warn("Failed to send chunk acknowledgment",
			"error", err,
			"chunk", chunkNumber)
		// Continue processing - non-fatal error
	}

	slog.Debug("Received tensor chunk",
		"model_id", modelID,
		"partition_id", partitionID,
		"tensor_id", tensorID,
		"chunk", chunkNumber,
		"compressed_size", len(data),
		"decompressed_size", len(decompressedData))

	return nil
}

// handleTensorStreamComplete handles tensor stream completion messages
func (sp *StreamingProtocol) handleTensorStreamComplete(header *StreamingHeader, data []byte) error {
	// Extract info from header
	modelID := header.ModelID
	partitionID := header.PartitionID
	tensorID := header.Header.TensorID

	// Generate stream ID
	streamID := GenerateStreamID(modelID, partitionID, tensorID)

	// Update transfer state
	sp.cacheMutex.Lock()
	defer sp.cacheMutex.Unlock()

	ts, exists := sp.transferCache[streamID]
	if !exists {
		return fmt.Errorf("transfer %s not found", streamID)
	}

	ts.Status = "complete"
	ts.LastUpdated = time.Now()
	ts.TotalChunks = header.TotalChunks

	slog.Info("Tensor transfer complete",
		"model_id", modelID,
		"partition_id", partitionID,
		"tensor_id", tensorID,
		"stream_id", streamID,
		"total_chunks", ts.TotalChunks,
		"bytes_transferred", ts.BytesTransferred)

	return nil
}

// handleTensorStreamRequest handles tensor stream request messages
func (sp *StreamingProtocol) handleTensorStreamRequest(header *StreamingHeader, data []byte) error {
	// In a real implementation, this would locate the requested tensor and stream it back
	// For this example, we'll log the request
	slog.Info("Received tensor request",
		"model_id", header.ModelID,
		"partition_id", header.PartitionID,
		"tensor_id", header.Header.TensorID,
		"priority", header.Priority)

	// Respond with a no-data message for now
	// In a real implementation, the server would begin streaming the tensor
	return nil
}

// handleTensorStreamResume handles tensor stream resume messages
func (sp *StreamingProtocol) handleTensorStreamResume(header *StreamingHeader, data []byte) error {
	// In a real implementation, this would resume streaming from the specified chunk
	// For this example, we'll log the resume request
	slog.Info("Received tensor resume request",
		"model_id", header.ModelID,
		"partition_id", header.PartitionID,
		"tensor_id", header.Header.TensorID,
		"chunk", header.ChunkNumber,
		"priority", header.Priority)

	// Respond with a no-data message for now
	// In a real implementation, the server would begin streaming from the specified chunk
	return nil
}

// handleTensorPrefetchRequest handles tensor prefetch request messages
func (sp *StreamingProtocol) handleTensorPrefetchRequest(header *StreamingHeader, data []byte) error {
	// In a real implementation, this would queue the tensor for low-priority streaming
	// For this example, we'll log the prefetch request
	slog.Info("Received tensor prefetch request",
		"model_id", header.ModelID,
		"partition_id", header.PartitionID,
		"tensor_id", header.Header.TensorID,
		"priority", header.Priority)

	// Respond with a no-data message for now
	// In a real implementation, the server would queue the tensor for streaming
	return nil
}

// handleTensorStreamCancel handles tensor stream cancel messages
func (sp *StreamingProtocol) handleTensorStreamCancel(header *StreamingHeader, data []byte) error {
	// Extract info from header
	modelID := header.ModelID
	partitionID := header.PartitionID
	tensorID := header.Header.TensorID

	// Generate stream ID
	streamID := GenerateStreamID(modelID, partitionID, tensorID)

	// Update transfer state
	sp.cacheMutex.Lock()
	defer sp.cacheMutex.Unlock()

	ts, exists := sp.transferCache[streamID]
	if !exists {
		return fmt.Errorf("transfer %s not found", streamID)
	}

	ts.Status = "cancelled"
	ts.LastUpdated = time.Now()

	slog.Info("Tensor transfer cancelled",
		"model_id", modelID,
		"partition_id", partitionID,
		"tensor_id", tensorID,
		"stream_id", streamID)

	return nil
}

// handleTensorHeartbeat handles heartbeat messages
func (sp *StreamingProtocol) handleTensorHeartbeat(header *StreamingHeader, data []byte) error {
	// Send heartbeat response
	responseHeader := StreamingHeader{
		Header: struct {
			Type StreamingMessageType
			MessageID uint32
			CorrelationID uint32
			Timestamp uint64
			TensorID string
			Size uint64
		}{
			Type:          TypeTensorHeartbeat,
			MessageID:     uint32(time.Now().UnixNano()),
			CorrelationID: header.Header.MessageID,
			Timestamp:     uint64(time.Now().Unix()),
			TensorID:      "",
			Size:          0, // No data in this message
		},
		ChunkNumber:     0,
		TotalChunks:     0,
		Priority:        0,
		CompressedSize:  0,
		Checksum:        [32]byte{},
		CompressionType: CompressionNone,
		ModelID:         header.ModelID,
		PartitionID:     header.PartitionID,
		Flags:           0,
	}

	return sp.SendStreamingMessage(responseHeader, nil)
}

// GetTransferState returns the current state of a transfer
func (sp *StreamingProtocol) GetTransferState(streamID string) (*TransferState, error) {
	sp.cacheMutex.RLock()
	defer sp.cacheMutex.RUnlock()

	ts, exists := sp.transferCache[streamID]
	if !exists {
		return nil, fmt.Errorf("transfer %s not found", streamID)
	}

	return ts, nil
}

// CancelTransfer cancels an in-progress transfer
func (sp *StreamingProtocol) CancelTransfer(streamID string) error {
	// Get transfer state
	sp.cacheMutex.RLock()
	ts, exists := sp.transferCache[streamID]
	sp.cacheMutex.RUnlock()

	if !exists {
		return fmt.Errorf("transfer %s not found", streamID)
	}

	// Create cancel request header
	cancelHeader := StreamingHeader{
		Header: struct {
			Type StreamingMessageType
			MessageID uint32
			CorrelationID uint32
			Timestamp uint64
			TensorID string
			Size uint64
		}{
			Type:          TypeTensorStreamCancel,
			MessageID:     uint32(time.Now().UnixNano()),
			CorrelationID: 0,
			Timestamp:     uint64(time.Now().Unix()),
			TensorID:      ts.TensorID,
			Size:          0, // No data in this message
		},
		ChunkNumber:     0,
		TotalChunks:     0,
		Priority:        uint8(ts.Priority),
		CompressedSize:  0,
		Checksum:        [32]byte{},
		CompressionType: CompressionNone,
		ModelID:         ts.ModelID,
		PartitionID:     ts.PartitionID,
		Flags:           0,
	}

	// Send the cancel request
	if err := sp.SendStreamingMessage(cancelHeader, nil); err != nil {
		return fmt.Errorf("failed to send cancel request: %w", err)
	}

	// Update transfer state
	sp.cacheMutex.Lock()
	if ts, exists := sp.transferCache[streamID]; exists {
		ts.Status = "cancelled"
		ts.LastUpdated = time.Now()
	}
	sp.cacheMutex.Unlock()

	slog.Info("Cancelled tensor transfer",
		"stream_id", streamID,
		"tensor_id", ts.TensorID,
		"model_id", ts.ModelID)

	return nil
}

// GetAllTransfers returns all transfers in the cache
func (sp *StreamingProtocol) GetAllTransfers() map[string]*TransferState {
	sp.cacheMutex.RLock()
	defer sp.cacheMutex.RUnlock()

	// Create a copy of the map to avoid concurrent modification issues
	transfers := make(map[string]*TransferState, len(sp.transferCache))
	for id, state := range sp.transferCache {
		transfers[id] = state
	}

	return transfers
}

// SubscribeTensor subscribes to tensor updates
func (sp *StreamingProtocol) SubscribeTensor(modelID, partitionID, tensorID string) error {
	// Create subscribe header
	subscribeHeader := StreamingHeader{
		Header: struct {
			Type StreamingMessageType
			MessageID uint32
			CorrelationID uint32
			Timestamp uint64
			TensorID string
			Size uint64
		}{
			Type:          TypeTensorSubscribe,
			MessageID:     uint32(time.Now().UnixNano()),
			CorrelationID: 0,
			Timestamp:     uint64(time.Now().Unix()),
			TensorID:      tensorID,
			Size:          0, // No data in this message
		},
		ChunkNumber:     0,
		TotalChunks:     0,
		Priority:        uint8(sp.defaultPriority),
		CompressedSize:  0,
		Checksum:        [32]byte{},
		CompressionType: CompressionNone,
		ModelID:         modelID,
		PartitionID:     partitionID,
		Flags:           0,
	}

	// Send the subscribe request
	if err := sp.SendStreamingMessage(subscribeHeader, nil); err != nil {
		return fmt.Errorf("failed to send tensor subscription: %w", err)
	}

	slog.Info("Subscribed to tensor updates",
		"model_id", modelID,
		"partition_id", partitionID,
		"tensor_id", tensorID)

	return nil
}

// SendHeartbeat sends a heartbeat message to keep the connection alive
func (sp *StreamingProtocol) SendHeartbeat() error {
	heartbeatHeader := StreamingHeader{
		Header: struct {
			Type StreamingMessageType
			MessageID uint32
			CorrelationID uint32
			Timestamp uint64
			TensorID string
			Size uint64
		}{
			Type:          TypeTensorHeartbeat,
			MessageID:     uint32(time.Now().UnixNano()),
			CorrelationID: 0,
			Timestamp:     uint64(time.Now().Unix()),
			TensorID:      "",
			Size:          0, // No data in this message
		},
		ChunkNumber:     0,
		TotalChunks:     0,
		Priority:        0,
		CompressedSize:  0,
		Checksum:        [32]byte{},
		CompressionType: CompressionNone,
		ModelID:         "",
		PartitionID:     "",
		Flags:           0,
	}

	return sp.SendStreamingMessage(heartbeatHeader, nil)
}

// StartHeartbeatRoutine starts sending periodic heartbeats
func (sp *StreamingProtocol) StartHeartbeatRoutine(ctx context.Context, interval time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				if err := sp.SendHeartbeat(); err != nil {
					slog.Warn("Failed to send heartbeat", "error", err)
				}
			case <-ctx.Done():
				return
			}
		}
	}()
}

// DecompressData decompresses data using the specified compression type
func (sp *StreamingProtocol) DecompressData(data []byte, compressionType CompressionType, originalSize uint64) ([]byte, error) {
	if compressionType == CompressionNone {
		return data, nil
	}
	
	return sp.compressor.Decompress(data, compressionType, originalSize)
}

// generateMessageID creates a unique message ID
func (sp *StreamingProtocol) generateMessageID() uint32 {
	sp.idMu.Lock()
	defer sp.idMu.Unlock()
	
	id := sp.nextMessageID
	sp.nextMessageID++
	return id
}

// SetReadDeadline sets a deadline for read operations
func (sp *StreamingProtocol) SetReadDeadline(t time.Time) error {
	return sp.conn.SetReadDeadline(t)
}

// SetWriteDeadline sets a deadline for write operations
func (sp *StreamingProtocol) SetWriteDeadline(t time.Time) error {
	return sp.conn.SetWriteDeadline(t)
}

// Close closes the streaming protocol and releases resources
func (sp *StreamingProtocol) Close() error {
	// Stop the cleanup routine
	if sp.stopCleanup != nil {
		close(sp.stopCleanup)
	}

	// Close the connection
	return sp.conn.Close()
}

// SendTensorRequest sends a request for a specific tensor
// Standard protocol compatibility method
func (sp *StreamingProtocol) SendTensorRequest(modelID, partitionID, tensorID string) error {
	// Use the StreamingProtocol's RequestTensor method
	_, err := sp.RequestTensor(modelID, partitionID, tensorID, sp.defaultPriority)
	return err
}

// SendTensorResponse sends a tensor in response to a request
// Standard protocol compatibility method
func (sp *StreamingProtocol) SendTensorResponse(requestID uint64, modelID, partitionID, tensorID string, data []byte) error {
	// For larger data, use the streaming approach
	if len(data) > sp.chunkSize {
		return sp.StreamTensor(modelID, partitionID, tensorID, data)
	}
	
	// Create a header for a single-chunk response
	header := StreamingHeader{
		Header: struct {
			Type StreamingMessageType
			MessageID uint32
			CorrelationID uint32
			Timestamp uint64
			TensorID string
			Size uint64
		}{
			Type:          TypeTensorStreamChunk,
			MessageID:     uint32(requestID), // Use the provided requestID
			CorrelationID: uint32(requestID),
			Timestamp:     uint64(time.Now().Unix()),
			TensorID:      tensorID,
			Size:          uint64(len(data)),
		},
		ChunkNumber:     0,
		TotalChunks:     1,
		Priority:        uint8(sp.defaultPriority),
		CompressedSize:  uint64(len(data)), // No compression for compatibility mode
		Checksum:        CalculateChecksum(data),
		CompressionType: CompressionNone,
		ModelID:         modelID,
		PartitionID:     partitionID,
		Flags:           0,
	}

	// Send the data as a single chunk
	return sp.SendStreamingMessage(header, data)
}

// SendTensorSync broadcasts a tensor update to other nodes
// Standard protocol compatibility method
func (sp *StreamingProtocol) SendTensorSync(modelID, partitionID, tensorID string, data []byte) error {
	// For larger data, use the streaming approach
	if len(data) > sp.chunkSize {
		return sp.StreamTensor(modelID, partitionID, tensorID, data)
	}
	
	// Create a header for a single-chunk sync
	header := StreamingHeader{
		Header: struct {
			Type StreamingMessageType
			MessageID uint32
			CorrelationID uint32
			Timestamp uint64
			TensorID string
			Size uint64
		}{
			Type:          TypeTensorSync, // Use the standard sync type for compatibility
			MessageID:     uint32(time.Now().UnixNano()),
			CorrelationID: 0,
			Timestamp:     uint64(time.Now().Unix()),
			TensorID:      tensorID,
			Size:          uint64(len(data)),
		},
		ChunkNumber:     0,
		TotalChunks:     1,
		Priority:        uint8(sp.defaultPriority),
		CompressedSize:  uint64(len(data)), // No compression for compatibility mode
		Checksum:        CalculateChecksum(data),
		CompressionType: CompressionNone,
		ModelID:         modelID,
		PartitionID:     partitionID,
		Flags:           0,
	}

	// Send the data as a single chunk
	return sp.SendStreamingMessage(header, data)
}

// SendAck sends an acknowledgment for a received message
// Standard protocol compatibility method
func (sp *StreamingProtocol) SendAck(requestID uint64, modelID, partitionID, tensorID string) error {
	// Create a header for an acknowledgment
	header := StreamingHeader{
		Header: struct {
			Type StreamingMessageType
			MessageID uint32
			CorrelationID uint32
			Timestamp uint64
			TensorID string
			Size uint64
		}{
			Type:          TypeTensorAck, // Use the standard ack type for compatibility
			MessageID:     uint32(time.Now().UnixNano()),
			CorrelationID: uint32(requestID),
			Timestamp:     uint64(time.Now().Unix()),
			TensorID:      tensorID,
			Size:          0, // No data in ack messages
		},
		ChunkNumber:     0,
		TotalChunks:     0,
		Priority:        uint8(sp.defaultPriority),
		CompressedSize:  0,
		Checksum:        [32]byte{},
		CompressionType: CompressionNone,
		ModelID:         modelID,
		PartitionID:     partitionID,
		Flags:           0,
	}

	// Send the ack message
	return sp.SendStreamingMessage(header, nil)
}

// SendError sends an error message
// Standard protocol compatibility method
func (sp *StreamingProtocol) SendError(requestID uint64, code int, errorMsg string) error {
	// Create a header for an error message
	header := StreamingHeader{
		Header: struct {
			Type StreamingMessageType
			MessageID uint32
			CorrelationID uint32
			Timestamp uint64
			TensorID string
			Size uint64
		}{
			Type:          TypeError, // Use the standard error type for compatibility
			MessageID:     uint32(time.Now().UnixNano()),
			CorrelationID: uint32(requestID),
			Timestamp:     uint64(time.Now().Unix()),
			TensorID:      "",
			Size:          uint64(len(errorMsg)),
		},
		ChunkNumber:     0,
		TotalChunks:     0,
		Priority:        uint8(sp.defaultPriority),
		CompressedSize:  0,
		Checksum:        [32]byte{},
		CompressionType: CompressionNone,
		ModelID:         "",
		PartitionID:     "",
		Flags:           uint32(code), // Store error code in flags
	}

	// Send the error message
	return sp.SendStreamingMessage(header, []byte(errorMsg))
}

// EncodeHeader serializes a streaming header into bytes
func EncodeHeader(header *StreamingHeader) []byte {
	// Create a buffer to hold the serialized header
	buf := new(bytes.Buffer)
	
	// Write the header fields
	binary.Write(buf, binary.LittleEndian, uint8(header.Header.Type))
	binary.Write(buf, binary.LittleEndian, header.Header.MessageID)
	binary.Write(buf, binary.LittleEndian, header.Header.CorrelationID)
	binary.Write(buf, binary.LittleEndian, header.Header.Timestamp)
	binary.Write(buf, binary.LittleEndian, header.Header.Size)
	
	// Write variable-length strings with their lengths
	writeString(buf, header.Header.TensorID)
	writeString(buf, header.ModelID)
	writeString(buf, header.PartitionID)
	
	// Write the streaming-specific fields
	binary.Write(buf, binary.LittleEndian, header.ChunkNumber)
	binary.Write(buf, binary.LittleEndian, header.TotalChunks)
	binary.Write(buf, binary.LittleEndian, header.Priority)
	binary.Write(buf, binary.LittleEndian, header.CompressedSize)
	binary.Write(buf, binary.LittleEndian, header.Checksum)
	binary.Write(buf, binary.LittleEndian, uint8(header.CompressionType))
	binary.Write(buf, binary.LittleEndian, header.Flags)
	
	return buf.Bytes()
}

// writeString writes a length-prefixed string to a buffer
func writeString(buf *bytes.Buffer, s string) {
	binary.Write(buf, binary.LittleEndian, uint16(len(s)))
	buf.WriteString(s)
}

// DecodeHeader deserializes bytes into a streaming header
func DecodeHeader(data []byte) (*StreamingHeader, error) {
	if len(data) < 4 {
		return nil, fmt.Errorf("header data too short")
	}
	
	buf := bytes.NewReader(data)
	header := &StreamingHeader{}
	
	// Read the header fields
	var msgType uint8
	binary.Read(buf, binary.LittleEndian, &msgType)
	header.Header.Type = StreamingMessageType(msgType)
	binary.Read(buf, binary.LittleEndian, &header.Header.MessageID)
	binary.Read(buf, binary.LittleEndian, &header.Header.CorrelationID)
	binary.Read(buf, binary.LittleEndian, &header.Header.Timestamp)
	binary.Read(buf, binary.LittleEndian, &header.Header.Size)
	
	// Read variable-length strings
	header.Header.TensorID = readString(buf)
	header.ModelID = readString(buf)
	header.PartitionID = readString(buf)
	
	// Read the streaming-specific fields
	binary.Read(buf, binary.LittleEndian, &header.ChunkNumber)
	binary.Read(buf, binary.LittleEndian, &header.TotalChunks)
	binary.Read(buf, binary.LittleEndian, &header.Priority)
	binary.Read(buf, binary.LittleEndian, &header.CompressedSize)
	binary.Read(buf, binary.LittleEndian, &header.Checksum)
	
	var compType uint8
	binary.Read(buf, binary.LittleEndian, &compType)
	header.CompressionType = CompressionType(compType)
	
	binary.Read(buf, binary.LittleEndian, &header.Flags)
	
	return header, nil
}

// readString reads a length-prefixed string from a reader
func readString(r *bytes.Reader) string {
	var length uint16
	binary.Read(r, binary.LittleEndian, &length)
	
	if length == 0 {
		return ""
	}
	
	str := make([]byte, length)
	r.Read(str)
	return string(str)
}

// CalculateChecksum computes a SHA-256 checksum for the given data
func CalculateChecksum(data []byte) [32]byte {
	return sha256.Sum256(data)
}

// GenerateStreamID creates a unique identifier for a tensor stream
func GenerateStreamID(modelID, partitionID, tensorID string) string {
	return fmt.Sprintf("%s:%s:%s:%d", modelID, partitionID, tensorID, time.Now().UnixNano())
}

// getIPType returns the type of IP address for logging/debugging
func getIPType(addr string) string {
	if addr == "unknown" || addr == "" {
		return "unknown"
	}
	
	// Extract just the IP part if there's a port
	ipStr := addr
	if idx := strings.LastIndex(addr, ":"); idx != -1 {
		ipStr = addr[:idx]
	}
	
	// Check common patterns
	switch {
	case ipStr == "127.0.0.1" || ipStr == "::1":
		return "loopback"
	case strings.HasPrefix(ipStr, "10.") || strings.HasPrefix(ipStr, "192.168.") ||
	     (strings.HasPrefix(ipStr, "172.") && func() bool {
			if len(ipStr) < 7 {
				return false
			}
			second, err := strconv.Atoi(ipStr[4:strings.Index(ipStr[4:], ".")+4])
			return err == nil && second >= 16 && second <= 31
		}()):
		return "private"
	case ipStr == "0.0.0.0" || ipStr == "::":
		return "any_interface"
	default:
		return "public"
	}
}
