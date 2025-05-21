package server

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
	
	"github.com/ollama/ollama/cluster"
	"github.com/ollama/ollama/cluster/model"
	"github.com/ollama/ollama/cluster/tensor"
	"log/slog"
)

// Global model transfer components
var (
	streamingServer     *StreamingTransferServer // Streaming-based transfer server
)

// StreamingTransferServer wraps the new streaming capabilities
type StreamingTransferServer struct {
	// Base directory for models
	modelsDir string
	
	// Server address (host:port)
	address string
	
	// Transfer manager for model transfers
	transferManager *model.ModelTransferManager
	
	// Listener for incoming connections
	listener net.Listener
	
	// Server context and cancellation
	ctx    context.Context
	cancel context.CancelFunc
	
	// State tracking
	running bool
	lock    sync.RWMutex
	
	// Map of active transfers
	activeTransfers map[string]model.TransferProgress
	transfersLock  sync.RWMutex
}

// InitializeModelTransferServer initializes and starts the streaming model transfer server
func InitializeModelTransferServer() error {
	// Check if already initialized
	if streamingServer != nil {
		slog.Info("Streaming model transfer server already initialized, reusing existing instance")
		return nil
	}

	// Determine models directory
	var modelsDir string
	if runtime.GOOS == "windows" {
		modelsDir = filepath.Join(os.Getenv("USERPROFILE"), ".ollama", "models")
	} else {
		modelsDir = filepath.Join(os.Getenv("HOME"), ".ollama", "models")
	}
	
	// Validate the models directory exists
	if _, err := os.Stat(modelsDir); os.IsNotExist(err) {
		slog.Warn("Models directory does not exist, creating it", "path", modelsDir)
		if err := os.MkdirAll(modelsDir, 0755); err != nil {
			slog.Error("Failed to create models directory", "error", err)
			return fmt.Errorf("failed to create models directory: %w", err)
		}
	}
	
	// Set up address for streaming protocol that matches the API port
	// This ensures the client connects to the same port for both API and streaming
	streamingAddr := "0.0.0.0:11434"
	
	slog.Info("Using API port for streaming protocol to ensure consistent connectivity",
		"streaming_address", streamingAddr)
	
	slog.Info("Starting streaming model transfer server",
		"streaming_address", streamingAddr,
		"models_dir", modelsDir,
		"os", runtime.GOOS)
	
	// Initialize streaming server
	initErr := initializeStreamingTransferServer(modelsDir, streamingAddr)
	if initErr != nil {
		slog.Error("Failed to initialize streaming transfer server",
			"error", initErr,
			"error_type", fmt.Sprintf("%T", initErr),
			"streaming_addr", streamingAddr)
		return fmt.Errorf("failed to initialize streaming transfer server: %w", initErr)
	}
	
	if streamingServer == nil {
		return fmt.Errorf("failed to initialize streaming transfer server - server is nil after initialization")
	}
	
	slog.Info("Streaming model transfer server started successfully",
		"address", streamingAddr,
		"running", streamingServer.running,
		"listener_open", streamingServer.listener != nil)
	
	return nil
}


// initializeStreamingTransferServer initializes the new streaming transfer server
func initializeStreamingTransferServer(modelsDir, address string) error {
	// Enhanced logging for initialization
	slog.Info("Beginning streaming transfer server initialization",
		"address", address,
		"models_dir", modelsDir,
		"os", runtime.GOOS,
		"pid", os.Getpid())
		
	// Check if models directory exists and is accessible
	dirInfo, err := os.Stat(modelsDir)
	if err != nil {
		if os.IsNotExist(err) {
			slog.Error("Models directory does not exist",
				"path", modelsDir,
				"error", err)
			// Try to create it
			if mkdirErr := os.MkdirAll(modelsDir, 0755); mkdirErr != nil {
				return fmt.Errorf("models directory does not exist and cannot be created: %w", mkdirErr)
			}
			slog.Info("Created models directory", "path", modelsDir)
		} else {
			slog.Error("Error accessing models directory",
				"path", modelsDir,
				"error", err)
			return fmt.Errorf("error accessing models directory: %w", err)
		}
	} else if !dirInfo.IsDir() {
		slog.Error("Models path exists but is not a directory", "path", modelsDir)
		return fmt.Errorf("models path exists but is not a directory: %s", modelsDir)
	}
	
	// Add retry logic for port binding with better error diagnostics
	maxRetries := 5 // Increased from 3 to 5
	var listener net.Listener
	var lastErr error
	
	// Check if port is available before attempting to bind
	conn, connErr := net.DialTimeout("tcp", address, 500*time.Millisecond)
	if connErr == nil {
		// Port is already in use by something that accepted our connection
		conn.Close()
		slog.Error("Port is already in use by another service that accepted a connection",
			"address", address)
		
		// On Windows, suggest netstat command
		if runtime.GOOS == "windows" {
			slog.Error("Suggested diagnostic command",
				"command", fmt.Sprintf("netstat -ano | findstr %s", strings.Split(address, ":")[1]))
		} else {
			slog.Error("Suggested diagnostic command",
				"command", fmt.Sprintf("lsof -i:%s", strings.Split(address, ":")[1]))
		}
		
		// Try an alternative port as fallback
		alternateAddress := "0.0.0.0:11435"
		slog.Warn("Attempting to use alternate port", "alternate_address", alternateAddress)
		address = alternateAddress
	}
	
	for i := 0; i < maxRetries; i++ {
		// Check if port is already in use
		listener, lastErr = net.Listen("tcp", address)
		if lastErr == nil {
			slog.Info("Successfully bound to port", "address", address, "attempt", i+1)
			break // Successfully bound to port
		}
		
		// Log detailed error information
		slog.Error("Failed to bind to streaming port",
			"address", address,
			"attempt", i+1,
			"max_attempts", maxRetries,
			"error", lastErr,
			"error_type", fmt.Sprintf("%T", lastErr),
			"error_message", lastErr.Error())
		
		// If this isn't the last retry, wait a moment and try again
		if i < maxRetries-1 {
			backoffMs := time.Duration((i+1)*1000) * time.Millisecond // Exponential backoff
			slog.Warn("Retrying port bind with backoff",
				"address", address,
				"attempt", i+1,
				"backoff_ms", backoffMs.Milliseconds())
			time.Sleep(backoffMs)
		}
	}
	
	// If we still have an error after retries, return it
	if lastErr != nil {
		slog.Error("All attempts to bind streaming transfer port failed",
			"address", address,
			"error", lastErr,
			"error_type", fmt.Sprintf("%T", lastErr))
		
		// Try to diagnose the problem more specifically
		if strings.Contains(lastErr.Error(), "address already in use") {
			// Check what's using the port
			if runtime.GOOS == "windows" {
				diagnosisCmd := fmt.Sprintf("netstat -ano | findstr %s", strings.Split(address, ":")[1])
				slog.Error("Port appears to be in use",
					"address", address,
					"diagnosis_command", diagnosisCmd)
			} else {
				diagnosisCmd := fmt.Sprintf("lsof -i:%s", strings.Split(address, ":")[1])
				slog.Error("Port appears to be in use",
					"address", address,
					"diagnosis_command", diagnosisCmd)
			}
			
			// Try another port as a last resort
			lastResortPort := "0.0.0.0:11437"
			slog.Warn("Attempting last resort port", "port", lastResortPort)
			if listener, lastErr = net.Listen("tcp", lastResortPort); lastErr == nil {
				address = lastResortPort
				slog.Info("Successfully bound to last resort port", "address", address)
			} else {
				return fmt.Errorf("streaming transfer port unavailable after %d attempts and fallback: %w", maxRetries, lastErr)
			}
		} else {
			return fmt.Errorf("streaming transfer port unavailable after %d attempts: %w", maxRetries, lastErr)
		}
	}
	
	// Create a new streaming transfer server with a more robust context
	ctx, cancel := context.WithCancel(context.Background())
	
	// Get the cluster registry from the cluster mode
	var registry *cluster.NodeRegistry
	if clusterMode2 != nil {
		registry = clusterMode2.Registry
		slog.Info("Using clusterMode2 registry for streaming server")
	} else {
		// If no registry is available, create a temporary one
		slog.Info("No existing registry found, creating temporary registry")
		registry = cluster.NewNodeRegistry(30*time.Second, 120*time.Second)
	}
	
	// Get the cluster configuration from the cluster mode
	var config *cluster.ClusterConfig
	if clusterMode2 != nil {
		config = clusterMode2.Config
		slog.Info("Using clusterMode2 configuration for streaming server")
	} else {
		// Use default configuration if cluster mode is not available
		slog.Info("No existing config found, using default cluster configuration")
		config = cluster.DefaultClusterConfig()
	}
	
	// Create the streaming transfer server
	streamingServer = &StreamingTransferServer{
		modelsDir:       modelsDir,
		address:         address,
		transferManager: model.NewModelTransferManager(registry, config),
		listener:        listener,
		ctx:             ctx,
		cancel:          cancel,
		running:         true,
		activeTransfers: make(map[string]model.TransferProgress),
	}
	
	// Validate initialization
	if streamingServer.transferManager == nil {
		slog.Error("Failed to initialize transfer manager",
			"address", address)
		return fmt.Errorf("transfer manager initialization failed")
	}
	
	// Start accepting connections in a background goroutine
	go streamingServer.acceptConnections()
	
	// Test that the server is actually accepting connections
	testConn, err := net.DialTimeout("tcp", address, 3*time.Second)
	if err != nil {
		slog.Error("Failed to verify streaming server is accepting connections",
			"address", address,
			"error", err)
		return fmt.Errorf("streaming server not accepting connections after initialization: %w", err)
	}
	testConn.Close()
	
	slog.Info("Streaming model transfer server started successfully",
		"address", address,
		"listener_open", streamingServer.listener != nil,
		"running", streamingServer.running)
	return nil
}

// acceptConnections handles incoming streaming protocol connections
func (s *StreamingTransferServer) acceptConnections() {
	for {
		// Check if server is still running
		s.lock.RLock()
		if !s.running {
			s.lock.RUnlock()
			return
		}
		s.lock.RUnlock()
		
		// Accept a connection with a timeout
		s.listener.(*net.TCPListener).SetDeadline(time.Now().Add(1 * time.Second))
		conn, err := s.listener.Accept()
		if err != nil {
			// Check if it's a timeout error (expected) or a real error
			if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
				// This is a timeout, just continue the loop
				continue
			}
			
			// Check if server is still running
			s.lock.RLock()
			running := s.running
			s.lock.RUnlock()
			
			if running {
				slog.Error("Error accepting connection", "error", err)
			}
			continue
		}
		
		// Handle the connection in a new goroutine
		go s.handleConnection(conn)
	}
}

// handleConnection processes an incoming connection
func (s *StreamingTransferServer) handleConnection(conn net.Conn) {
	defer conn.Close()
	
	// Enhanced logging with connection details
	remoteAddr := conn.RemoteAddr().String()
	localAddr := "unknown"
	if conn.LocalAddr() != nil {
		localAddr = conn.LocalAddr().String()
	}
	
	slog.Info("New streaming connection established",
		"remote_addr", remoteAddr,
		"local_addr", localAddr,
		"os", runtime.GOOS)
	
	// Windows-specific TCP optimizations for better stability
	if runtime.GOOS == "windows" {
		if tcpConn, ok := conn.(*net.TCPConn); ok {
			// Set larger receive buffer for Windows TCP connections
			if err := tcpConn.SetReadBuffer(4 * 1024 * 1024); err != nil {
				slog.Warn("Failed to set larger TCP read buffer on Windows",
					"error", err,
					"remote_addr", remoteAddr)
			}
			
			// Set larger send buffer for Windows TCP connections
			if err := tcpConn.SetWriteBuffer(4 * 1024 * 1024); err != nil {
				slog.Warn("Failed to set larger TCP write buffer on Windows",
					"error", err,
					"remote_addr", remoteAddr)
			}
			
			// Disable Nagle's algorithm for streaming performance
			if err := tcpConn.SetNoDelay(true); err != nil {
				slog.Warn("Failed to disable Nagle algorithm on Windows TCP connection",
					"error", err,
					"remote_addr", remoteAddr)
			}
			
			// Set keep-alive to detect dead connections
			if err := tcpConn.SetKeepAlive(true); err != nil {
				slog.Warn("Failed to enable TCP keep-alive on Windows connection",
					"error", err,
					"remote_addr", remoteAddr)
			} else {
				// Set keep-alive period to 30 seconds
				if err := tcpConn.SetKeepAlivePeriod(30 * time.Second); err != nil {
					slog.Warn("Failed to set TCP keep-alive period on Windows connection",
						"error", err,
						"remote_addr", remoteAddr)
				}
			}
			
			slog.Info("Applied Windows-specific TCP optimizations",
				"remote_addr", remoteAddr,
				"buffer_size", "4MB")
		}
	}
	
	// Create a streaming protocol instance for this connection
	streamingProto := tensor.NewStreamingProtocol(conn)
	
	// Process messages
	for {
		// Set a read deadline to prevent hanging
		conn.SetReadDeadline(time.Now().Add(30 * time.Second))
		
		// Receive a message
		header, data, err := streamingProto.ReceiveStreamingMessage()
		if err != nil {
			// Check if it's a normal EOF or a real error
			if err.Error() == "EOF" || err.Error() == "connection closed" {
				slog.Info("Connection closed by client", 
					"remote_addr", conn.RemoteAddr().String())
			} else {
				slog.Error("Error receiving message", 
					"error", err, 
					"remote_addr", conn.RemoteAddr().String())
			}
			return
		}
		
		// Handle the message
		err = streamingProto.HandleStreamingMessage(header, data)
		if err != nil {
			slog.Error("Error handling streaming message", 
				"error", err, 
				"message_type", header.Header.Type,
				"remote_addr", conn.RemoteAddr().String())
			
			// Don't return, try to continue processing messages
		}
	}
}

// TransferModel uses the streaming protocol to transfer a model
func (s *StreamingTransferServer) TransferModel(ctx context.Context, modelName, sourcePath string, destNodeID string) (string, error) {
	// Add extensive diagnostic logging
	slog.Info("Starting model transfer with streaming protocol",
		"model_name", modelName,
		"source_path", sourcePath,
		"dest_node", destNodeID,
		"streaming_enabled", s != nil,
		"transfer_manager", s.transferManager != nil,
		"protocol", "streaming")
	
	// Validate model path exists
	_, err := os.Stat(sourcePath)
	if err != nil {
		slog.Error("Model path validation failed",
			"error", err,
			"source_path", sourcePath,
			"model", modelName)
		return "", fmt.Errorf("model source path validation failed: %w", err)
	}
	
	// Validate destination node is valid
	if destNodeID == "" {
		slog.Error("Invalid destination node ID", "node_id", destNodeID)
		return "", fmt.Errorf("destination node ID cannot be empty")
	}
	
	// Create a transfer request
	request := model.TransferRequest{
		ModelID:          modelName,
		PartitionID:      "default",  // Use default partition for single transfers
		SourceNodeID:     "local",    // Source is local node
		DestinationNodeID: destNodeID,
		Operation:        model.TransferOperationPush,
		Priority:         tensor.PriorityHigh,
		Mode:             "streaming",
	}
	
	slog.Info("Created transfer request",
		"request", fmt.Sprintf("%+v", request),
		"model", modelName)
	
	// Use the transfer manager to initiate the transfer
	transferID, err := s.transferManager.TransferTensors(ctx, request)
	if err != nil {
		slog.Error("Failed to initiate model transfer",
			"error", err,
			"error_type", fmt.Sprintf("%T", err),
			"model", modelName,
			"dest_node", destNodeID)
		return "", fmt.Errorf("failed to initiate model transfer: %w", err)
	}
	
	// Register callback to track transfer progress
	s.transferManager.RegisterTransferCallback(s.updateTransferProgress)
	
	slog.Info("Transfer initiated successfully",
		"transfer_id", transferID,
		"model", modelName,
		"dest_node", destNodeID)
	
	return transferID, nil
}

// updateTransferProgress tracks transfer progress
func (s *StreamingTransferServer) updateTransferProgress(transferID string, progress model.TransferProgress) {
	s.transfersLock.Lock()
	defer s.transfersLock.Unlock()
	
	s.activeTransfers[transferID] = progress
	
	// Log progress updates
	percentComplete := 0
	if progress.TotalBytes > 0 {
		percentComplete = int((progress.BytesTransferred * 100) / progress.TotalBytes)
	}
	
	slog.Info("Model transfer progress", 
		"transfer_id", transferID,
		"state", progress.State,
		"percent_complete", percentComplete,
		"bytes_transferred", progress.BytesTransferred,
		"total_bytes", progress.TotalBytes)
	
	// Clean up completed or failed transfers after a while
	if progress.State == model.TransferStateCompleted || 
	   progress.State == model.TransferStateFailed ||
	   progress.State == model.TransferStateCancelled {
		
		// Start a goroutine to remove this after a delay
		go func(id string) {
			time.Sleep(1 * time.Hour)
			s.transfersLock.Lock()
			delete(s.activeTransfers, id)
			s.transfersLock.Unlock()
		}(transferID)
	}
}

// GetTransferProgress retrieves the progress of a specific transfer
func (s *StreamingTransferServer) GetTransferProgress(transferID string) (model.TransferProgress, error) {
	s.transfersLock.RLock()
	defer s.transfersLock.RUnlock()
	
	progress, exists := s.activeTransfers[transferID]
	if !exists {
		return model.TransferProgress{}, fmt.Errorf("transfer %s not found", transferID)
	}
	
	return progress, nil
}

// Stop stops the streaming transfer server
func (s *StreamingTransferServer) Stop() {
	s.lock.Lock()
	defer s.lock.Unlock()
	
	if !s.running {
		return
	}
	
	s.running = false
	
	// Cancel context and close listener
	if s.cancel != nil {
		s.cancel()
	}
	
	if s.listener != nil {
		s.listener.Close()
	}
	
	// Close transfer manager
	if s.transferManager != nil {
		s.transferManager.Close()
	}
	
	slog.Info("Streaming transfer server stopped")
}

// ShutdownModelTransferServer stops the streaming model transfer server
func ShutdownModelTransferServer() {
	// Stop streaming server if running
	if streamingServer != nil {
		slog.Info("Stopping streaming model transfer server")
		streamingServer.Stop()
		streamingServer = nil
	}
}


// GetStreamingTransferServer returns the streaming transfer server
func GetStreamingTransferServer() *StreamingTransferServer {
	return streamingServer
}

// TransferModelWithStreaming uses the streaming protocol to transfer a model
// and is the main entry point for external components
func TransferModelWithStreaming(ctx context.Context, modelName string, sourcePath string, destNodeID string) (string, error) {
	// Add comprehensive logging for diagnostics with stack trace for better debugging
	defer func() {
		if r := recover(); r != nil {
			stack := make([]byte, 8192)
			stack = stack[:runtime.Stack(stack, false)]
			slog.Error("PANIC in TransferModelWithStreaming",
				"error", fmt.Sprintf("%v", r),
				"model", modelName,
				"stack", string(stack))
		}
	}()

	slog.Info("TransferModelWithStreaming called",
		"model", modelName,
		"source_path", sourcePath,
		"dest_node", destNodeID,
		"streaming_server_initialized", streamingServer != nil,
		"api_port", "11434",  // API port for reference
		"time", time.Now())
		
	if streamingServer == nil {
		// Enhanced diagnostics about streaming server state
		slog.Error("Streaming server not initialized - detailed diagnostics",
			"model", modelName,
			"time", time.Now().Format(time.RFC3339),
			"goroutines", runtime.NumGoroutine(),
			"os", runtime.GOOS,
			"pid", os.Getpid(),
			"client_api_port", "11434") // Main API port
			
		// Capture current stack trace for debugging
		stack := make([]byte, 8192)
		stack = stack[:runtime.Stack(stack, false)]
		slog.Info("Current stack trace at transfer attempt", "stack", string(stack))
		
		// Check for ports in use that might indicate existing server
		checkPorts := []string{"11434", "11435", "11437"}
		for _, port := range checkPorts {
			// Try to detect if the port is already in use
			conn, err := net.DialTimeout("tcp", "127.0.0.1:"+port, 500*time.Millisecond)
			if err == nil {
				conn.Close()
				slog.Warn("Port detected as already in use", "port", port)
			} else {
				slog.Info("Port appears available", "port", port)
			}
		}
			
		// Try to initialize the streaming server on demand with extended timeout
		initCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		
		// Run initialization in a separate goroutine with timeout
		initErrCh := make(chan error, 1)
		go func() {
			initErr := InitializeModelTransferServer()
			initErrCh <- initErr
		}()
		
		// Wait for initialization or timeout
		var err error
		select {
		case err = <-initErrCh:
			// Initialization completed (success or error)
		case <-initCtx.Done():
			err = fmt.Errorf("timeout waiting for streaming server initialization")
			slog.Error("Streaming server initialization timed out", "timeout", "30s")
		}
		
		if err != nil {
			slog.Error("Failed to initialize streaming server on demand",
				"error", err,
				"error_type", fmt.Sprintf("%T", err),
				"error_msg", err.Error())
			
			// More comprehensive diagnostic information
			if os.IsPermission(err) {
				slog.Error("Permission denied error detected - check directory permissions")
			} else if os.IsNotExist(err) {
				slog.Error("Path does not exist error detected - check directory paths")
			} else if strings.Contains(err.Error(), "address already in use") {
				slog.Error("Port conflict detected - another process may be using the streaming port")
				
				// Try to identify what's using the port
				if runtime.GOOS == "windows" {
					cmd := exec.Command("netstat", "-ano")
					output, _ := cmd.CombinedOutput()
					slog.Error("Port usage info (netstat)", "output", string(output))
				} else {
					cmd := exec.Command("lsof", "-i", ":11434")
					output, _ := cmd.CombinedOutput()
					slog.Error("Port usage info (lsof)", "output", string(output))
				}
			}
			
			return "", fmt.Errorf("failed to initialize streaming server with enhanced diagnostics: %w", err)
		}
		
		// Check if initialization was successful with more detailed validation
		if streamingServer == nil {
			slog.Error("Streaming transfer server still nil after initialization attempt - critical configuration error")
			return "", fmt.Errorf("critical configuration error: streaming transfer server still nil after initialization attempt")
		}
		
		// Additional verification of server state
		if !streamingServer.running || streamingServer.listener == nil {
			slog.Error("Streaming server initialized but in invalid state",
				"running", streamingServer != nil && streamingServer.running,
				"listener_nil", streamingServer == nil || streamingServer.listener == nil,
				"address", func() string {
					if streamingServer != nil {
						return streamingServer.address
					}
					return "unknown"
				}())
			return "", fmt.Errorf("streaming server initialized but in invalid state (running=%v, listener_nil=%v)",
				streamingServer != nil && streamingServer.running,
				streamingServer == nil || streamingServer.listener == nil)
		}
		
		slog.Info("Successfully initialized streaming server on demand",
			"server_address", streamingServer.address,
			"running", streamingServer.running,
			"tcp_listener_valid", streamingServer.listener != nil,
			"transfer_manager_valid", streamingServer.transferManager != nil)
	}
	
	// Validate server state with more robust diagnostics and recovery options
	if !streamingServer.running {
		slog.Error("Streaming server not in running state - attempting recovery",
			"model", modelName,
			"server_address", streamingServer.address,
			"server_state", fmt.Sprintf("%+v", streamingServer),
			"time", time.Now().Format(time.RFC3339))
			
		// Attempt emergency restart of streaming server
		slog.Info("Attempting emergency restart of streaming server")
		ShutdownModelTransferServer() // First shut down any existing server
		
		// Try reinitialization with timeout
		initCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		
		initErrCh := make(chan error, 1)
		go func() {
			initErr := InitializeModelTransferServer()
			initErrCh <- initErr
		}()
		
		// Wait for initialization or timeout
		var restartErr error
		select {
		case restartErr = <-initErrCh:
			// Initialization completed (success or error)
		case <-initCtx.Done():
			restartErr = fmt.Errorf("timeout waiting for streaming server restart")
		}
		
		if restartErr != nil {
			slog.Error("Emergency restart of streaming server failed",
				"error", restartErr)
			return "", fmt.Errorf("streaming server not in running state and emergency restart failed: %w", restartErr)
		}
		
		if streamingServer == nil || !streamingServer.running {
			return "", fmt.Errorf("streaming server still not running after emergency restart")
		}
		
		slog.Info("Emergency restart of streaming server successful",
			"server_address", streamingServer.address)
	}
	
	// Test if the listener is actually available with enhanced diagnostics
	if streamingServer.listener == nil {
		slog.Error("Streaming server listener is nil - critical error",
			"model", modelName,
			"server_address", streamingServer.address,
			"running_state", streamingServer.running,
			"transfer_manager_valid", streamingServer.transferManager != nil)
			
		// Try recreation of listener as a recovery measure
		addr := streamingServer.address
		slog.Info("Attempting to recreate listener", "address", addr)
		
		newListener, err := net.Listen("tcp", addr)
		if err != nil {
			slog.Error("Failed to recreate listener",
				"address", addr,
				"error", err)
			return "", fmt.Errorf("streaming server listener is nil and recreation failed: %w", err)
		}
		
		// Update the server with new listener
		streamingServer.listener = newListener
		slog.Info("Successfully recreated listener", "address", addr)
		
		// Start accepting connections again
		go streamingServer.acceptConnections()
	}
	
	// Test connectivity to our own streaming server with comprehensive retry logic
	var testConn net.Conn
	var err error
	var connSuccess bool
	
	// Try multiple times with increasing timeouts
	timeouts := []time.Duration{500 * time.Millisecond, 1 * time.Second, 3 * time.Second, 5 * time.Second}
	for _, timeout := range timeouts {
		slog.Info("Attempting connectivity test to streaming server",
			"attempt_timeout", timeout,
			"address", streamingServer.address)
			
		testConn, err = net.DialTimeout("tcp", streamingServer.address, timeout)
		if err == nil {
			connSuccess = true
			testConn.Close()
			slog.Info("Successfully connected to streaming server",
				"attempt_timeout", timeout,
				"address", streamingServer.address)
			break
		}
		
		slog.Warn("Connection test attempt failed",
			"timeout", timeout,
			"address", streamingServer.address,
			"error", err)
	}
	
	if !connSuccess {
		// Try alternate addresses in case the configured one is wrong
		alternateAddresses := []string{
			"127.0.0.1:11434",
			"localhost:11434",
			"0.0.0.0:11434",
			"127.0.0.1:11435",
		}
		
		for _, addr := range alternateAddresses {
			if addr == streamingServer.address {
				continue // Skip the one we already tried
			}
			
			slog.Info("Trying alternate address for connectivity test", "address", addr)
			altConn, altErr := net.DialTimeout("tcp", addr, 3*time.Second)
			
			if altErr == nil {
				// Found working address
				altConn.Close()
				slog.Info("Found working alternate address for streaming server",
					"original_address", streamingServer.address,
					"working_address", addr)
				
				// Update streaming server with working address
				streamingServer.address = addr
				connSuccess = true
				break
			}
		}
	}
	
	if !connSuccess {
		// If we still can't connect, log detailed diagnostics and return error
		slog.Error("Cannot connect to streaming server after multiple attempts",
			"address", streamingServer.address,
			"error", err,
			"error_type", fmt.Sprintf("%T", err),
			"os", runtime.GOOS,
			"network_diagnostic", "connection_failed")
		
		return "", fmt.Errorf("cannot connect to streaming server after multiple retry attempts: %w", err)
	}
	
	slog.Info("Successfully verified connectivity to streaming server with enhanced reliability checks")
	
	// Validate inputs before passing to transfer function
	if modelName == "" {
		return "", fmt.Errorf("model name cannot be empty")
	}
	
	if sourcePath == "" {
		return "", fmt.Errorf("source path cannot be empty")
	}
	
	if destNodeID == "" {
		return "", fmt.Errorf("destination node ID cannot be empty")
	}
	
	// Verify model file exists
	_, err = os.Stat(sourcePath)
	if err != nil {
		if os.IsNotExist(err) {
			slog.Error("Model file does not exist",
				"path", sourcePath,
				"model", modelName)
			return "", fmt.Errorf("model file does not exist: %w", err)
		}
		slog.Error("Error checking model file",
			"path", sourcePath,
			"error", err)
		return "", fmt.Errorf("error checking model file: %w", err)
	}
	
	// Log additional diagnostic information about destination
	slog.Info("Preparing transfer with enhanced diagnostics",
		"model", modelName,
		"source_path", sourcePath,
		"dest_node", destNodeID,
		"transfer_manager", streamingServer.transferManager != nil,
		"server_address", streamingServer.address)
	
	// Check connectivity to streaming server
	conn, err := net.DialTimeout("tcp", streamingServer.address, 5*time.Second)
	if err != nil {
		slog.Error("Failed to connect to streaming server",
			"address", streamingServer.address,
			"error", err)
		return "", fmt.Errorf("cannot connect to streaming server at %s: %w", streamingServer.address, err)
	}
	conn.Close()
	slog.Info("Successfully verified connectivity to streaming server",
		"address", streamingServer.address)
	
	// Call the transfer function with detailed error handling
	transferID, err := streamingServer.TransferModel(ctx, modelName, sourcePath, destNodeID)
	if err != nil {
		slog.Error("Streaming transfer failed",
			"error", err,
			"error_type", fmt.Sprintf("%T", err),
			"model", modelName,
			"dest_node", destNodeID,
			"traceback", fmt.Sprintf("%+v", err))
		return "", fmt.Errorf("streaming transfer failed: %w", err)
	}
	
	slog.Info("Transfer initiated successfully",
		"transfer_id", transferID,
		"model", modelName,
		"dest_node", destNodeID)
	
	return transferID, nil
}

// IsStreamingEnabled checks if streaming transfers are available
func IsStreamingEnabled() bool {
	return streamingServer != nil
}