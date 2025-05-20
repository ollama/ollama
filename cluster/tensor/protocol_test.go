package tensor

import (
	"bytes"
	"encoding/binary"
	"math/rand"
	"testing"
)

// Mock tensor data for testing
func createMockTensorData(size int, dtype TensorDataType) []byte {
	data := make([]byte, size)
	rand.Read(data) // Fill with random bytes
	return data
}

func TestCreateTensorMessage(t *testing.T) {
	// Test parameters
	id := uint32(42)
	operation := TensorOperationTransfer
	dtype := TensorDataTypeFloat32
	shape := []int{2, 3, 4} // 2x3x4 tensor
	tensorData := createMockTensorData(2*3*4*4, dtype) // 4 bytes per float32
	
	// Create message
	msg := CreateTensorMessage(id, operation, dtype, shape, tensorData)
	
	// Verify fields
	if msg.Header.MessageID != id {
		t.Errorf("Expected message ID %d, got %d", id, msg.Header.MessageID)
	}
	
	if msg.Header.Operation != operation {
		t.Errorf("Expected operation %d, got %d", operation, msg.Header.Operation)
	}
	
	if msg.Header.DataType != dtype {
		t.Errorf("Expected data type %d, got %d", dtype, msg.Header.DataType)
	}
	
	if len(msg.Header.Shape) != len(shape) {
		t.Errorf("Expected shape length %d, got %d", len(shape), len(msg.Header.Shape))
	}
	
	for i, dim := range shape {
		if msg.Header.Shape[i] != dim {
			t.Errorf("Expected shape[%d] = %d, got %d", i, dim, msg.Header.Shape[i])
		}
	}
	
	if !bytes.Equal(msg.Data, tensorData) {
		t.Error("Tensor data mismatch")
	}
	
	// Verify protocol version
	if msg.Header.ProtocolVersion != CurrentProtocolVersion {
		t.Errorf("Expected protocol version %d, got %d", 
			CurrentProtocolVersion, msg.Header.ProtocolVersion)
	}
}

func TestSerializeDeserialize(t *testing.T) {
	// Create a test message
	original := &TensorMessage{
		Header: TensorMessageHeader{
			ProtocolVersion: CurrentProtocolVersion,
			MessageID:       123,
			Operation:       TensorOperationTransfer,
			DataType:        TensorDataTypeFloat32,
			Shape:           []int{2, 3},
			OriginalSize:    24, // 2*3*4 bytes
			CompressedSize:  24,
			Flags:           0,
		},
		Data: createMockTensorData(24, TensorDataTypeFloat32),
	}
	
	// Serialize
	data, err := original.Serialize()
	if err != nil {
		t.Fatalf("Failed to serialize message: %v", err)
	}
	
	// Deserialize
	deserialized, err := DeserializeTensorMessage(data)
	if err != nil {
		t.Fatalf("Failed to deserialize message: %v", err)
	}
	
	// Verify fields match
	if deserialized.Header.MessageID != original.Header.MessageID {
		t.Errorf("MessageID mismatch: %d vs %d", 
			original.Header.MessageID, deserialized.Header.MessageID)
	}
	
	if deserialized.Header.Operation != original.Header.Operation {
		t.Errorf("Operation mismatch: %d vs %d", 
			original.Header.Operation, deserialized.Header.Operation)
	}
	
	if deserialized.Header.DataType != original.Header.DataType {
		t.Errorf("DataType mismatch: %d vs %d", 
			original.Header.DataType, deserialized.Header.DataType)
	}
	
	if len(deserialized.Header.Shape) != len(original.Header.Shape) {
		t.Errorf("Shape length mismatch: %d vs %d", 
			len(original.Header.Shape), len(deserialized.Header.Shape))
	} else {
		for i := range original.Header.Shape {
			if deserialized.Header.Shape[i] != original.Header.Shape[i] {
				t.Errorf("Shape[%d] mismatch: %d vs %d", 
					i, original.Header.Shape[i], deserialized.Header.Shape[i])
			}
		}
	}
func TestIncompatibleProtocolVersion(t *testing.T) {
	// Create a buffer that starts with an incompatible version
	buffer := new(bytes.Buffer)
	
	// Write incompatible protocol version
	incompatibleVersion := byte(CurrentProtocolVersion + 10)
	buffer.WriteByte(incompatibleVersion)
	
	// Add dummy data to complete the header
	dummyHeader := make([]byte, 64)
	buffer.Write(dummyHeader)
	
	// Try to deserialize
	_, err := DeserializeTensorMessage(buffer.Bytes())
	
	// Should fail with protocol version error
	if err == nil {
		t.Error("Expected error for incompatible protocol version, but got nil")
	}
	
	if err != ErrIncompatibleProtocolVersion {
		t.Errorf("Expected ErrIncompatibleProtocolVersion, got %v", err)
	}
}

func TestCompression(t *testing.T) {
	// Create tensor with compressible data (e.g., many zeros)
	tensorSize := 1024
	tensorData := make([]byte, tensorSize)
	// Leave as zeros to ensure good compression
	
	// Create message
	original := CreateTensorMessage(
		1, 
		TensorOperationTransfer, 
		TensorDataTypeFloat32,
		[]int{256, 1}, 
		tensorData,
	)
	
	// Set compression flag
	original.Header.Flags |= TensorFlagCompressed
	
	// Apply compression
	err := original.Compress(CompressionLevelDefault)
	if err != nil {
		t.Fatalf("Failed to compress: %v", err)
	}
	
	// Verify compressed size is smaller than original
	if original.Header.CompressedSize >= uint32(tensorSize) {
		t.Error("Expected compression to reduce data size")
	}
	
	// Serialize
	data, err := original.Serialize()
	if err != nil {
		t.Fatalf("Failed to serialize compressed message: %v", err)
	}
	
	// Deserialize
	deserialized, err := DeserializeTensorMessage(data)
	if err != nil {
		t.Fatalf("Failed to deserialize compressed message: %v", err)
	}
	
	// Decompress
	err = deserialized.Decompress()
	if err != nil {
		t.Fatalf("Failed to decompress: %v", err)
	}
	
	// Verify decompressed data matches original
	if len(deserialized.Data) != tensorSize {
		t.Errorf("Expected decompressed size %d, got %d", tensorSize, len(deserialized.Data))
	}
	
	if !bytes.Equal(deserialized.Data, tensorData) {
		t.Error("Data mismatch after compression/decompression cycle")
	}
}

func TestCorruptedData(t *testing.T) {
	// Create a valid message
	original := CreateTensorMessage(
		1, 
		TensorOperationTransfer, 
		TensorDataTypeFloat32,
		[]int{2, 2}, 
		createMockTensorData(16, TensorDataTypeFloat32),
	)
	
	// Serialize it
	data, err := original.Serialize()
	if err != nil {
		t.Fatalf("Failed to serialize: %v", err)
	}
	
	// Corrupt the size field in the header
	// Find a good spot in the header to corrupt that won't affect fundamental parsing
	if len(data) > 20 {
		// Corrupt a byte in the header
		data[20] = ^data[20] // Flip all bits
	}
	
	// Try to deserialize
	_, err = DeserializeTensorMessage(data)
	
	// Should fail with some error
	if err == nil {
		t.Error("Expected error for corrupted data, but got nil")
	}
	
	// Corrupt the data portion
	data, _ = original.Serialize()
	if len(data) > 100 {
		// Corrupt a byte in the data section
		data[len(data)-10] = ^data[len(data)-10]
	}
	
	// This should still deserialize, but data integrity is compromised
	deserialized, err := DeserializeTensorMessage(data)
	if err != nil {
		t.Fatalf("Unexpected error with corrupted data: %v", err)
	}
	
	// Data should not match original
	if bytes.Equal(deserialized.Data, original.Data) {
		t.Error("Expected data mismatch after corruption, but data matches")
	}
}

func TestBatchOperations(t *testing.T) {
	// Test batch operation (e.g., for grouped transfers)
	original := CreateTensorMessage(
		100,
		TensorOperationBatchBegin,
		TensorDataTypeNone,
		[]int{}, // No shape for control messages
		nil,     // No data for control messages
	)
	
	// Set batch ID and size
	batchID := uint32(42)
	batchSize := uint16(5)
	
	// Encode batch info into the message
	buffer := new(bytes.Buffer)
	binary.Write(buffer, binary.LittleEndian, batchID)
	binary.Write(buffer, binary.LittleEndian, batchSize)
	original.Data = buffer.Bytes()
	
	// Serialize
	data, err := original.Serialize()
	if err != nil {
		t.Fatalf("Failed to serialize batch message: %v", err)
	}
	
	// Deserialize
	deserialized, err := DeserializeTensorMessage(data)
	if err != nil {
		t.Fatalf("Failed to deserialize batch message: %v", err)
	}
	
	// Verify operation
	if deserialized.Header.Operation != TensorOperationBatchBegin {
		t.Errorf("Expected operation %d, got %d", 
			TensorOperationBatchBegin, deserialized.Header.Operation)
	}
	
	// Decode batch info
	var decodedBatchID uint32
	var decodedBatchSize uint16
	buffer = bytes.NewBuffer(deserialized.Data)
	binary.Read(buffer, binary.LittleEndian, &decodedBatchID)
	binary.Read(buffer, binary.LittleEndian, &decodedBatchSize)
	
	// Verify batch info
	if decodedBatchID != batchID {
		t.Errorf("Expected batch ID %d, got %d", batchID, decodedBatchID)
	}
	
	if decodedBatchSize != batchSize {
		t.Errorf("Expected batch size %d, got %d", batchSize, decodedBatchSize)
	}
}
	
	if !bytes.Equal(deserialized.Data, original.Data) {
		t.Error("Data mismatch after serialization/deserialization")
	}
}