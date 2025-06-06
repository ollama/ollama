package gguf

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestRead(t *testing.T) {
	// Setup
	tempDir := t.TempDir()
	tempFile := filepath.Join(tempDir, "test.gguf")

	if err := createTestGGUFFile(tempFile, map[string]any{
		"general.architecture": "llama",
		"general.alignment":    int64(32),
	}, []testTensorInfo{
		{Name: "token_embd.weight", Shape: []uint64{1000, 512}, Type: 1}, // F16
		{Name: "output.weight", Shape: []uint64{512, 1000}, Type: 1},     // F16
	}); err != nil {
		t.Fatal(err)
	}

	f, err := Open(tempFile)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	// Test
	if got := f.NumKeyValues(); got != 2 {
		t.Errorf("NumKeyValues() = %d, want %d", got, 2)
	}
	if got := f.NumTensors(); got != 2 {
		t.Errorf("NumTensors() = %d, want %d", got, 2)
	}
	archKV := f.KeyValue("general.architecture")
	if archKV.Key == "" {
		t.Error("KeyValue(\"general.architecture\") not found")
	}
	if got := archKV.String(); got != "llama" {
		t.Errorf("KeyValue(\"general.architecture\").String() = %q, want %q", got, "llama")
	}
	alignKV := f.KeyValue("general.alignment")
	if alignKV.Key == "" {
		t.Error("KeyValue(\"general.alignment\") not found")
	}
	if got := alignKV.Int(); got != 32 {
		t.Errorf("KeyValue(\"general.alignment\").Int() = %d, want %d", got, 32)
	}
	expectedTensorNames := []string{"token_embd.weight", "output.weight"}
	var gotTensorNames []string
	for _, tensor := range f.TensorInfos() {
		gotTensorNames = append(gotTensorNames, tensor.Name)
	}
	if !slices.Equal(gotTensorNames, expectedTensorNames) {
		t.Errorf("tensor names = %v, want %v", gotTensorNames, expectedTensorNames)
	}
	tokenTensor := f.TensorInfo("token_embd.weight")
	if tokenTensor.Name != "token_embd.weight" {
		t.Error("TensorInfo(\"token_embd.weight\") not found")
	}
	if len(tokenTensor.Shape) == 0 {
		t.Error("TensorInfo(\"token_embd.weight\") has empty shape")
	}
	outputTensor := f.TensorInfo("output.weight")
	if outputTensor.Name != "output.weight" {
		t.Error("TensorInfo(\"output.weight\") not found")
	}
	if len(outputTensor.Shape) == 0 {
		t.Error("TensorInfo(\"output.weight\") has empty shape")
	}
	var gotKeyCount int
	for _, kv := range f.KeyValues() {
		gotKeyCount++
		if kv.Key == "" {
			t.Error("found key value with empty key")
		}
	}
	if gotKeyCount != 2 {
		t.Errorf("iterated key count = %d, want %d", gotKeyCount, 2)
	}
	tensorInfo, reader, err := f.TensorReader("token_embd.weight")
	if err != nil {
		t.Errorf("TensorReader(\"token_embd.weight\") error: %v", err)
	}
	if tensorInfo.Name != "token_embd.weight" {
		t.Errorf("TensorReader returned wrong tensor: %q", tensorInfo.Name)
	}
	if reader == nil {
		t.Error("TensorReader returned nil reader")
	}
}

func BenchmarkRead(b *testing.B) {
	// Create benchmark test file
	tempDir := b.TempDir()
	tempFile := filepath.Join(tempDir, "benchmark.gguf")

	if err := createTestGGUFFile(tempFile, map[string]any{
		"general.architecture": "llama",
		"general.alignment":    int64(32),
	}, []testTensorInfo{
		{Name: "token_embd.weight", Shape: []uint64{1000, 512}, Type: 1}, // F16
		{Name: "output.weight", Shape: []uint64{512, 1000}, Type: 1},     // F16
	}); err != nil {
		b.Fatal(err)
	}

	// Get file info for reporting
	info, err := os.Stat(tempFile)
	if err != nil {
		b.Fatal(err)
	}
	b.Logf("Benchmark file size: %d bytes", info.Size())

	b.ReportAllocs()

	for b.Loop() {
		f, err := Open(tempFile)
		if err != nil {
			b.Fatal(err)
		}

		// Access some data to ensure it's actually being read
		_ = f.KeyValue("general.architecture").String()
		_ = f.KeyValue("general.alignment").Int()
		_ = f.NumTensors()
		_ = f.NumKeyValues()

		// Iterate through some tensors
		count := 0
		for _, tensor := range f.TensorInfos() {
			_ = tensor.Name
			count++
			if count >= 2 {
				break
			}
		}

		f.Close()
	}
}

// Helper function to create test GGUF files
func createTestGGUFFile(path string, keyValues map[string]any, tensors []testTensorInfo) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	// Write GGUF magic
	if _, err := file.Write([]byte("GGUF")); err != nil {
		return err
	}

	// Write version
	if err := binary.Write(file, binary.LittleEndian, uint32(3)); err != nil {
		return err
	}

	// Write tensor count
	if err := binary.Write(file, binary.LittleEndian, uint64(len(tensors))); err != nil {
		return err
	}

	// Write metadata count
	if err := binary.Write(file, binary.LittleEndian, uint64(len(keyValues))); err != nil {
		return err
	}

	// Write metadata
	for key, value := range keyValues {
		if err := writeKeyValue(file, key, value); err != nil {
			return err
		}
	}

	// Write tensor info
	for _, tensor := range tensors {
		if err := writeTensorInfo(file, tensor); err != nil {
			return err
		}
	}

	// Write some dummy tensor data
	dummyData := make([]byte, 1024)
	file.Write(dummyData)

	return nil
}

type testTensorInfo struct {
	Name  string
	Shape []uint64
	Type  uint32
}

func writeKeyValue(file *os.File, key string, value any) error {
	// Write key length and key
	if err := binary.Write(file, binary.LittleEndian, uint64(len(key))); err != nil {
		return err
	}
	if _, err := file.Write([]byte(key)); err != nil {
		return err
	}

	// Write value based on type
	switch v := value.(type) {
	case string:
		if err := binary.Write(file, binary.LittleEndian, uint32(typeString)); err != nil {
			return err
		}
		if err := binary.Write(file, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		_, err := file.Write([]byte(v))
		return err
	case int64:
		if err := binary.Write(file, binary.LittleEndian, typeInt64); err != nil {
			return err
		}
		return binary.Write(file, binary.LittleEndian, v)
	case bool:
		if err := binary.Write(file, binary.LittleEndian, typeBool); err != nil {
			return err
		}
		return binary.Write(file, binary.LittleEndian, v)
	case float64:
		if err := binary.Write(file, binary.LittleEndian, uint32(typeFloat64)); err != nil {
			return err
		}
		return binary.Write(file, binary.LittleEndian, v)
	case []string:
		if err := binary.Write(file, binary.LittleEndian, uint32(typeArray)); err != nil {
			return err
		}
		if err := binary.Write(file, binary.LittleEndian, typeString); err != nil {
			return err
		}
		if err := binary.Write(file, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		for _, s := range v {
			if err := binary.Write(file, binary.LittleEndian, uint64(len(s))); err != nil {
				return err
			}
			if _, err := file.Write([]byte(s)); err != nil {
				return err
			}
		}
		return nil
	case []int64:
		if err := binary.Write(file, binary.LittleEndian, uint32(typeArray)); err != nil {
			return err
		}
		if err := binary.Write(file, binary.LittleEndian, typeInt64); err != nil {
			return err
		}
		if err := binary.Write(file, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		for _, i := range v {
			if err := binary.Write(file, binary.LittleEndian, i); err != nil {
				return err
			}
		}
		return nil
	case []float64:
		if err := binary.Write(file, binary.LittleEndian, typeArray); err != nil {
			return err
		}
		if err := binary.Write(file, binary.LittleEndian, typeFloat64); err != nil {
			return err
		}
		if err := binary.Write(file, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		for _, f := range v {
			if err := binary.Write(file, binary.LittleEndian, f); err != nil {
				return err
			}
		}
		return nil
	default:
		return fmt.Errorf("unsupported value type: %T", value)
	}
}

func writeTensorInfo(file *os.File, tensor testTensorInfo) error {
	// Write tensor name
	if err := binary.Write(file, binary.LittleEndian, uint64(len(tensor.Name))); err != nil {
		return err
	}
	if _, err := file.Write([]byte(tensor.Name)); err != nil {
		return err
	}

	// Write dimensions
	if err := binary.Write(file, binary.LittleEndian, uint32(len(tensor.Shape))); err != nil {
		return err
	}
	for _, dim := range tensor.Shape {
		if err := binary.Write(file, binary.LittleEndian, dim); err != nil {
			return err
		}
	}

	// Write type
	if err := binary.Write(file, binary.LittleEndian, tensor.Type); err != nil {
		return err
	}

	// Write offset (dummy value)
	return binary.Write(file, binary.LittleEndian, uint64(0))
}
