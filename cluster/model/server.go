package model

import (
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"sync"
	"time"
	
	"github.com/ollama/ollama/cluster/tensor"
	"log/slog"
)

// TransferServer listens for model transfer requests
type TransferServer struct {
	listener     net.Listener
	modelsDir    string
	activeJobs   map[string]*TransferJob
	activeJobsMu sync.RWMutex
	stopCh       chan struct{}
	wg           sync.WaitGroup
}

// TransferJob represents an active model transfer job
type TransferJob struct {
	ModelName  string
	StartTime  time.Time
	BytesTotal int64
	BytesRecv  int64
	Status     string
	Error      error
}

// NewTransferServer creates a new model transfer server
func NewTransferServer(modelsDir string, listenAddr string) (*TransferServer, error) {
	// Create listener
	listener, err := net.Listen("tcp", listenAddr)
	if err != nil {
		return nil, fmt.Errorf("failed to create model transfer listener: %w", err)
	}
	
	server := &TransferServer{
		listener:   listener,
		modelsDir:  modelsDir,
		activeJobs: make(map[string]*TransferJob),
		stopCh:     make(chan struct{}),
	}
	
	return server, nil
}

// Start begins accepting connections for model transfers
func (s *TransferServer) Start() {
	s.wg.Add(1)
	go s.acceptLoop()
	
	slog.Info("Model transfer server started",
		"address", s.listener.Addr().String())
}

// Stop gracefully stops the server
func (s *TransferServer) Stop() {
	close(s.stopCh)
	s.listener.Close()
	s.wg.Wait()
	
	slog.Info("Model transfer server stopped")
}

// acceptLoop accepts and handles incoming connections
func (s *TransferServer) acceptLoop() {
	defer s.wg.Done()
	
	for {
		select {
		case <-s.stopCh:
			return
		default:
			// Continue accepting connections
		}
		
		// Set accept deadline to allow for periodic stop checks
		s.listener.(*net.TCPListener).SetDeadline(time.Now().Add(1 * time.Second))
		
		conn, err := s.listener.Accept()
		if err != nil {
			if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
				// This is just the periodic timeout, continue
				continue
			}
			
			select {
			case <-s.stopCh:
				// Server is shutting down, return gracefully
				return
			default:
				// Unexpected error
				slog.Error("Error accepting model transfer connection",
					"error", err)
				continue
			}
		}
		
		// Handle connection in a new goroutine
		s.wg.Add(1)
		go func() {
			defer s.wg.Done()
			s.handleConnection(conn)
		}()
	}
}

// handleConnection processes a model transfer connection
func (s *TransferServer) handleConnection(conn net.Conn) {
	defer conn.Close()
	
	remoteAddr := conn.RemoteAddr().String()
	slog.Info("New model transfer connection", "remote", remoteAddr)
	
	// Create streaming protocol handler
	proto := tensor.NewStreamingProtocol(conn)
	
	// Set a larger timeout for the initial message
	conn.SetReadDeadline(time.Now().Add(5 * time.Minute))
	
	// Receive the first message to determine what model is being transferred
	header, data, err := proto.ReceiveStreamingMessage()
	if err != nil {
		slog.Error("Failed to receive initial message",
			"error", err,
			"remote", remoteAddr)
		return
	}
	
	// Clear the deadline after initial message
	conn.SetReadDeadline(time.Time{})
	
	// Verify it's a tensor request or sync message
	if header.Header.Type != tensor.TypeTensorRequest && header.Header.Type != tensor.TypeTensorSync {
		slog.Error("Unexpected initial message type",
			"type", header.Header.Type,
			"remote", remoteAddr)
		
		// Send error response
		_ = proto.SendError(uint64(header.Header.MessageID), 400, "Expecting TensorRequest or TensorSync")
		return
	}
	
	// Extract model information
	modelName := header.ModelID
	if modelName == "" {
		slog.Error("Missing model name in initial message",
			"remote", remoteAddr)
		
		// Send error response
		_ = proto.SendError(uint64(header.Header.MessageID), 400, "Missing model name")
		return
	}
	
	// Create a job to track this transfer
	job := &TransferJob{
		ModelName: modelName,
		StartTime: time.Now(),
		Status:    "receiving",
	}
	
	// Register the job
	jobID := fmt.Sprintf("%s-%s-%d", modelName, remoteAddr, time.Now().UnixNano())
	s.activeJobsMu.Lock()
	s.activeJobs[jobID] = job
	s.activeJobsMu.Unlock()
	
	// Ensure job is removed when done
	defer func() {
		s.activeJobsMu.Lock()
		delete(s.activeJobs, jobID)
		s.activeJobsMu.Unlock()
	}()
	
	slog.Info("Starting model transfer receive",
		"model", modelName,
		"remote", remoteAddr,
		"job_id", jobID)
	
	// Handle based on message type
	switch header.Header.Type {
	case tensor.TypeTensorRequest:
		// Model request - client wants us to send a model
		// Not implemented in this version
		_ = proto.SendError(uint64(header.Header.MessageID), 501, "Model serving not implemented")
		return
		
	case tensor.TypeTensorSync:
		// Model sync - client is sending us a model
		err = s.handleModelSync(proto, job, header, data)
		if err != nil {
			slog.Error("Failed to handle model sync",
				"error", err,
				"model", modelName,
				"remote", remoteAddr)
			job.Status = "failed"
			job.Error = err
			return
		}
	}
	
	slog.Info("Model transfer completed successfully",
		"model", modelName,
		"remote", remoteAddr,
		"bytes_received", job.BytesRecv,
		"duration", time.Since(job.StartTime))
	
	job.Status = "completed"
}

// handleModelSync processes a model sync operation (receiving a model)
func (s *TransferServer) handleModelSync(proto *tensor.StreamingProtocol, job *TransferJob, firstHeader *tensor.StreamingHeader, firstData []byte) error {
	modelName := firstHeader.ModelID
	partitionID := firstHeader.PartitionID
	tensorID := firstHeader.Header.TensorID
	modelDir := filepath.Join(s.modelsDir, modelName)
	
	// Create model directory if it doesn't exist
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		return fmt.Errorf("failed to create model directory: %w", err)
	}
	
	// Create temporary file for the model
	tmpFilePath := filepath.Join(modelDir, fmt.Sprintf("model.%d.tmp", time.Now().UnixNano()))
	file, err := os.Create(tmpFilePath)
	if err != nil {
		return fmt.Errorf("failed to create temporary model file: %w", err)
	}
	
	// Ensure file is closed and cleanup on error
	success := false
	defer func() {
		file.Close()
		if !success {
			// Clean up the temporary file on failure
			os.Remove(tmpFilePath)
		}
	}()
	
	// Generate a transfer ID for tracking this operation
	transferID := tensor.GenerateStreamID(modelName, partitionID, tensorID)
	
	// Process first chunk if it was included
	if len(firstData) > 0 {
		if _, err = file.Write(firstData); err != nil {
			return fmt.Errorf("failed to write initial data: %w", err)
		}
		job.BytesRecv += int64(len(firstData))
	}
	
	// Send acknowledgment for the first message
	ackHeader := tensor.StreamingHeader{
		Header: struct {
			Type          tensor.StreamingMessageType
			MessageID     uint32
			CorrelationID uint32
			Timestamp     uint64
			TensorID      string
			Size          uint64
		}{
			Type:          tensor.TypeTensorStreamAck,
			MessageID:     uint32(time.Now().UnixNano()),
			CorrelationID: uint32(firstHeader.Header.MessageID),
			Timestamp:     uint64(time.Now().Unix()),
			TensorID:      tensorID,
			Size:          0,
		},
		ChunkNumber:     0,
		TotalChunks:     0,
		Priority:        uint8(tensor.PriorityNormal),
		CompressedSize:  0,
		Checksum:        [32]byte{},
		CompressionType: tensor.CompressionNone,
		ModelID:         modelName,
		PartitionID:     partitionID,
		Flags:           0,
	}
	
	if err = proto.SendStreamingMessage(ackHeader, nil); err != nil {
		return fmt.Errorf("failed to send initial ack: %w", err)
	}
	
	slog.Info("Starting model transfer using streaming protocol",
		"model", modelName,
		"partition", partitionID,
		"tensor", tensorID,
		"transfer_id", transferID)
	
	// Continue receiving chunks
	chunkCount := 0
	for {
		// Create a context with timeout to prevent hanging
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		
		// Set a read deadline on the connection
		deadline, _ := ctx.Deadline()
		proto.SetReadDeadline(deadline)
		
		// Use the streaming-specific receive method
		header, data, err := proto.ReceiveStreamingMessage()
		cancel() // Clean up the context right after use
		
		if err != nil {
			if err == io.EOF {
				// Clean end of transfer
				break
			}
			return fmt.Errorf("error receiving chunk: %w", err)
		}
		
		// Check if this is a completion message
		if header.Header.Type == tensor.TypeTensorStreamComplete {
			// This indicates the transfer is complete
			slog.Info("Received transfer completion message",
				"model", modelName,
				"total_chunks", chunkCount)
			break
		}
		
		// Verify message type
		if header.Header.Type != tensor.TypeTensorStreamChunk && header.Header.Type != tensor.TypeTensorSync {
			return fmt.Errorf("unexpected message type: %v", header.Header.Type)
		}
		
		// Decompress the data if needed
		var decompressedData []byte
		if header.CompressionType != tensor.CompressionNone {
			decompressed, err := proto.DecompressData(data, header.CompressionType, uint64(len(data)))
			if err != nil {
				return fmt.Errorf("failed to decompress chunk %d: %w", header.ChunkNumber, err)
			}
			decompressedData = decompressed
		} else {
			decompressedData = data
		}
		
		// Verify checksum
		if header.CompressionType != tensor.CompressionNone {
			calculatedChecksum := tensor.CalculateChecksum(decompressedData)
			if calculatedChecksum != header.Checksum {
				slog.Warn("Checksum mismatch for chunk",
					"chunk", header.ChunkNumber,
					"expected", header.Checksum,
					"calculated", calculatedChecksum)
				// Continue anyway - could implement retry logic here
			}
		}
		
		// Write the data to the file
		if len(decompressedData) > 0 {
			if _, err = file.Write(decompressedData); err != nil {
				return fmt.Errorf("failed to write chunk %d: %w", header.ChunkNumber, err)
			}
			job.BytesRecv += int64(len(decompressedData))
			
			slog.Debug("Received model chunk",
				"model", modelName,
				"chunk", header.ChunkNumber,
				"total_chunks", header.TotalChunks,
				"compressed_size", len(data),
				"decompressed_size", len(decompressedData),
				"total_received", job.BytesRecv)
			
			chunkCount++
		}
		
		// Send acknowledgment
		ackHeader := tensor.StreamingHeader{
			Header: struct {
				Type          tensor.StreamingMessageType
				MessageID     uint32
				CorrelationID uint32
				Timestamp     uint64
				TensorID      string
				Size          uint64
			}{
				Type:          tensor.TypeTensorStreamAck,
				MessageID:     uint32(time.Now().UnixNano()),
				CorrelationID: header.Header.MessageID,
				Timestamp:     uint64(time.Now().Unix()),
				TensorID:      header.Header.TensorID,
				Size:          0,
			},
			ChunkNumber:     header.ChunkNumber,
			TotalChunks:     header.TotalChunks,
			Priority:        header.Priority,
			CompressedSize:  0,
			Checksum:        [32]byte{},
			CompressionType: tensor.CompressionNone,
			ModelID:         header.ModelID,
			PartitionID:     header.PartitionID,
			Flags:           0,
		}
		
		if err = proto.SendStreamingMessage(ackHeader, nil); err != nil {
			slog.Warn("Failed to send chunk acknowledgment",
				"error", err,
				"chunk", header.ChunkNumber)
			// Continue processing - non-fatal error
		}
	}
	
	// Sync file to ensure all data is written
	if err = file.Sync(); err != nil {
		return fmt.Errorf("failed to sync file: %w", err)
	}
	
	// Close the file before renaming
	file.Close()
	
	// Move temporary file to final model file
	finalPath := filepath.Join(modelDir, "model")
	if err = os.Rename(tmpFilePath, finalPath); err != nil {
		return fmt.Errorf("failed to finalize model file: %w", err)
	}
	
	// Mark as success to prevent cleanup
	success = true
	
	// Update job status
	job.Status = "completed"
	
	return nil
}

// GetActiveJob retrieves information about an active transfer job
func (s *TransferServer) GetActiveJob(jobID string) (*TransferJob, bool) {
	s.activeJobsMu.RLock()
	defer s.activeJobsMu.RUnlock()
	
	job, exists := s.activeJobs[jobID]
	return job, exists
}

// GetAllActiveJobs returns all active transfer jobs
func (s *TransferServer) GetAllActiveJobs() []*TransferJob {
	s.activeJobsMu.RLock()
	defer s.activeJobsMu.RUnlock()
	
	jobs := make([]*TransferJob, 0, len(s.activeJobs))
	for _, job := range s.activeJobs {
		jobs = append(jobs, job)
	}
	return jobs
}