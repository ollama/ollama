package tensor

import (
	"bytes"
	"io"
	"math/rand"
	"net"
	"sync"
	"testing"
	"time"
)

// mockConn implements a mock net.Conn for testing
type mockConn struct {
	readBuf  *bytes.Buffer
	writeBuf *bytes.Buffer
	closed   bool
	mu       sync.Mutex
}

func newMockConn() *mockConn {
	return &mockConn{
		readBuf:  new(bytes.Buffer),
		writeBuf: new(bytes.Buffer),
	}
}

func (m *mockConn) Read(b []byte) (n int, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return 0, net.ErrClosed
	}
	return m.readBuf.Read(b)
}

func (m *mockConn) Write(b []byte) (n int, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return 0, net.ErrClosed
	}
	return m.writeBuf.Write(b)
}

func (m *mockConn) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	return nil
}

func (m *mockConn) LocalAddr() net.Addr                { return &net.TCPAddr{} }
func (m *mockConn) RemoteAddr() net.Addr               { return &net.TCPAddr{} }
func (m *mockConn) SetDeadline(t time.Time) error      { return nil }
func (m *mockConn) SetReadDeadline(t time.Time) error  { return nil }
func (m *mockConn) SetWriteDeadline(t time.Time) error { return nil }

// Helper to feed data to the mock connection
func (m *mockConn) FeedData(data []byte) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.readBuf.Write(data)
}

// Helper to get written data from the mock connection
func (m *mockConn) GetWrittenData() []byte {
	m.mu.Lock()
	defer m.mu.Unlock()
	data := m.writeBuf.Bytes()
	m.writeBuf.Reset()
	return data
}

// Tests for StreamingProtocol

func TestStreamingProtocolBasics(t *testing.T) {
	mockConn := newMockConn()
	proto := NewStreamingProtocol(mockConn)

	// Test sending a tensor request
	err := proto.SendTensorRequest("model1", "partition1", "tensor1")
	if err != nil {
		t.Fatalf("Failed to send tensor request: %v", err)
	}

	// Get written data
	data := mockConn.GetWrittenData()
	if len(data) == 0 {
		t.Fatal("No data was written")
	}

	// Feed the data back to create a response
	mockConn.FeedData(data)

	// Read the message
	header, receivedData, err := proto.ReceiveStreamingMessage()
	if err != nil {
		t.Fatalf("Failed to receive streaming message: %v", err)
	}

	// Verify header
	if header.Header.Type != TypeTensorStreamRequest {
		t.Errorf("Expected message type %d, got %d", TypeTensorStreamRequest, header.Header.Type)
	}

	if header.ModelID != "model1" {
		t.Errorf("Expected model ID 'model1', got '%s'", header.ModelID)
	}

	if header.PartitionID != "partition1" {
		t.Errorf("Expected partition ID 'partition1', got '%s'", header.PartitionID)
	}

	if header.Header.TensorID != "tensor1" {
		t.Errorf("Expected tensor ID 'tensor1', got '%s'", header.Header.TensorID)
	}

	// No data in request
	if len(receivedData) != 0 {
		t.Errorf("Expected empty data, got %d bytes", len(receivedData))
	}
}

func TestStreamingProtocolCompatibility(t *testing.T) {
	mockConn := newMockConn()
	proto := NewStreamingProtocol(mockConn)

	// Test sending data using standard protocol methods
	err := proto.SendTensorSync("model1", "partition1", "tensor1", []byte("test data"))
	if err != nil {
		t.Fatalf("Failed to send tensor sync: %v", err)
	}

	// Get written data
	data := mockConn.GetWrittenData()
	if len(data) == 0 {
		t.Fatal("No data was written")
	}

	// Feed the data back
	mockConn.FeedData(data)

	// Read using standard protocol method (which should use streaming internally)
	header, receivedData, err := proto.ReceiveMessage()
	if err != nil {
		t.Fatalf("Failed to receive message: %v", err)
	}

	// Verify header
	if header.Type != TypeTensorSync {
		t.Errorf("Expected message type %d, got %d", TypeTensorSync, header.Type)
	}

	if header.ModelID != "model1" {
		t.Errorf("Expected model ID 'model1', got '%s'", header.ModelID)
	}

	// Verify data
	if string(receivedData) != "test data" {
		t.Errorf("Expected data 'test data', got '%s'", string(receivedData))
	}
}

func TestStreamTensorChunking(t *testing.T) {
	mockConn := newMockConn()
	proto := NewStreamingProtocol(mockConn)

	// Set a small chunk size for testing
	proto.SetChunkSize(10) // 10 bytes per chunk

	// Create test data larger than chunk size
	testData := make([]byte, 25) // Will require 3 chunks
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	// Start a goroutine to read messages and send acks
	var receivedChunks []*StreamingHeader
	var receivedData [][]byte
	var wg sync.WaitGroup
	wg.Add(1)
	
	go func() {
		defer wg.Done()
		
		// Read info message
		header, data, err := proto.ReceiveStreamingMessage()
		if err != nil {
			t.Errorf("Failed to receive info message: %v", err)
			return
		}
		
		if header.Header.Type != TypeTensorStreamInfo {
			t.Errorf("Expected info message, got %d", header.Header.Type)
			return
		}
		
		// Read chunks and send acks
		for {
			header, data, err := proto.ReceiveStreamingMessage()
			if err != nil {
				if err == io.EOF {
					break
				}
				t.Errorf("Error receiving chunk: %v", err)
				return
			}
			
			receivedChunks = append(receivedChunks, header)
			receivedData = append(receivedData, data)
			
			// Send an ack
			ackHeader := StreamingHeader{
				Header: Header{
					Type:          TypeTensorStreamAck,
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
				CompressionType: CompressionNone,
				ModelID:         header.ModelID,
				PartitionID:     header.PartitionID,
				Flags:           0,
			}
			
			if err := proto.SendStreamingMessage(ackHeader, nil); err != nil {
				t.Errorf("Failed to send ack: %v", err)
				return
			}
			
			// If this is the last chunk, we're done
			if header.Header.Type == TypeTensorStreamComplete || 
			   (header.ChunkNumber == header.TotalChunks-1) {
				break
			}
		}
	}()
	
	// Send the data
	err := proto.StreamTensor("model1", "partition1", "tensor1", testData)
	if err != nil {
		t.Fatalf("Failed to stream tensor: %v", err)
	}
	
	// Wait for receiver goroutine to finish
	wg.Wait()
	
	// Verify chunks
	if len(receivedChunks) < 3 { // Should have at least 3 chunks (could have more with control messages)
		t.Errorf("Expected at least 3 chunks, got %d", len(receivedChunks))
	}
	
	// Reconstruct the data
	reconstructed := make([]byte, 0, len(testData))
	for _, data := range receivedData {
		if len(data) > 0 {
			reconstructed = append(reconstructed, data...)
		}
	}
	
	// Verify reconstructed data
	if !bytes.Equal(reconstructed, testData) {
		t.Error("Reconstructed data doesn't match original")
	}
}

func TestDecompressData(t *testing.T) {
	proto := NewStreamingProtocol(newMockConn())

	// Test with no compression
	testData := []byte("test data with no compression")
	result, err := proto.DecompressData(testData, CompressionNone, uint64(len(testData)))
	if err != nil {
		t.Fatalf("DecompressData failed with no compression: %v", err)
	}
	if !bytes.Equal(result, testData) {
		t.Error("DecompressData modified data with no compression")
	}

	// Create compressible data (repeated pattern)
	compressibleData := make([]byte, 1000)
	for i := range compressibleData {
		compressibleData[i] = byte(i % 10)
	}

	// Compress the data
	compressedData, _, err := proto.compressor.CompressWithStats(compressibleData, CompressionLZ4)
	if err != nil {
		t.Fatalf("Failed to compress data: %v", err)
	}

	// Decompress and verify
	decompressed, err := proto.DecompressData(compressedData, CompressionLZ4, uint64(len(compressibleData)))
	if err != nil {
		t.Fatalf("DecompressData failed with LZ4 compression: %v", err)
	}
	if !bytes.Equal(decompressed, compressibleData) {
		t.Error("DecompressData did not correctly decompress LZ4 data")
	}
}

func TestCompatibilityMethods(t *testing.T) {
	// Create a mock connection
	mockConn := newMockConn()
	proto := NewStreamingProtocol(mockConn)

	// Test each compatibility method
	testCases := []struct {
		name     string
		testFunc func() error
	}{
		{
			name: "SendTensorRequest",
			testFunc: func() error {
				return proto.SendTensorRequest("model1", "partition1", "tensor1")
			},
		},
		{
			name: "SendTensorResponse",
			testFunc: func() error {
				return proto.SendTensorResponse(123, "model1", "partition1", "tensor1", []byte("test data"))
			},
		},
		{
			name: "SendTensorSync",
			testFunc: func() error {
				return proto.SendTensorSync("model1", "partition1", "tensor1", []byte("test data"))
			},
		},
		{
			name: "SendAck",
			testFunc: func() error {
				return proto.SendAck(123, "model1", "partition1", "tensor1")
			},
		},
		{
			name: "SendError",
			testFunc: func() error {
				return proto.SendError(123, 404, "Test error message")
			},
		},
	}

	// Run each test case
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Reset buffers
			mockConn.writeBuf.Reset()
			mockConn.readBuf.Reset()

			// Execute test function
			err := tc.testFunc()
			if err != nil {
				t.Fatalf("%s failed: %v", tc.name, err)
			}

			// Verify data was written
			data := mockConn.GetWrittenData()
			if len(data) == 0 {
				t.Fatalf("%s did not write any data", tc.name)
			}

			// Feed data back to verify it can be read
			mockConn.FeedData(data)

			// Try to read it
			_, _, err = proto.ReceiveStreamingMessage()
			if err != nil {
				t.Fatalf("Failed to receive message for %s: %v", tc.name, err)
			}
		})
	}
}

func TestStreamingErrorHandling(t *testing.T) {
	// Create a mock connection
	mockConn := newMockConn()
	proto := NewStreamingProtocol(mockConn)

	// Test closing connection during transfer
	proto.SetChunkSize(10)
	testData := make([]byte, 100) // 10 chunks
	rand.Read(testData)

	// Start a goroutine to close the connection
	go func() {
		// Wait a bit to ensure transfer has started
		time.Sleep(10 * time.Millisecond)
		mockConn.Close()
	}()

	// This should fail because the connection is closed
	err := proto.StreamTensor("model1", "partition1", "tensor1", testData)
	if err == nil {
		t.Fatal("Expected error when connection closed during streaming, but got nil")
	}
}