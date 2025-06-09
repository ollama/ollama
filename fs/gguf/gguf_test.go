package gguf

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
)

func TestRead(t *testing.T) {
	// Setup
	tempDir := t.TempDir()
	tempFile := filepath.Join(tempDir, t.Name())

	// Create test file using WriteGGUF
	file, err := os.Create(tempFile)
	if err != nil {
		t.Fatal(err)
	}

	// Prepare key-value pairs
	kv := ggml.KV{
		"general.architecture": "llama",
		"general.alignment":    uint32(32),
	}

	// Prepare tensors with dummy data
	dummyData1 := make([]byte, 1000*512*2) // F16 = 2 bytes per element
	dummyData2 := make([]byte, 512*1000*2) // F16 = 2 bytes per element

	tensors := []*ggml.Tensor{
		{
			Name:     "token_embd.weight",
			Kind:     1, // F16
			Shape:    []uint64{1000, 512},
			WriterTo: bytes.NewReader(dummyData1),
		},
		{
			Name:     "output.weight",
			Kind:     1, // F16
			Shape:    []uint64{512, 1000},
			WriterTo: bytes.NewReader(dummyData2),
		},
	}

	if err := ggml.WriteGGUF(file, kv, tensors); err != nil {
		file.Close()
		t.Fatal(err)
	}
	file.Close()

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
	if got := archKV.String(); got != "llama" {
		t.Errorf("KeyValue(\"general.architecture\").String() = %q, want %q", got, "llama")
	}
	alignKV := f.KeyValue("general.alignment")
	if got := alignKV.Uint(); got != 32 {
		t.Errorf("KeyValue(\"general.alignment\").Int() = %d, want %d", got, 32)
	}

	expectedTensorNames := []string{"token_embd.weight", "output.weight"}
	var gotTensorNames []string
	outputTensor := f.TensorInfo("output.weight")
	if len(outputTensor.Shape) == 0 {
		t.Error("TensorInfo(\"output.weight\") has empty shape")
	}
	tokenTensor := f.TensorInfo("token_embd.weight")
	if len(tokenTensor.Shape) == 0 {
		t.Error("TensorInfo(\"token_embd.weight\") has empty shape")
	}
	for _, tensor := range f.TensorInfos() {
		gotTensorNames = append(gotTensorNames, tensor.Name)
	}
	if !slices.Equal(gotTensorNames, expectedTensorNames) {
		t.Errorf("tensor names = %v, want %v", gotTensorNames, expectedTensorNames)
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

	// Create test file using WriteGGUF
	file, err := os.Create(tempFile)
	if err != nil {
		b.Fatal(err)
	}

	// Prepare key-value pairs
	kv := ggml.KV{
		"general.architecture": "llama",
		"general.alignment":    uint32(32),
	}

	// Prepare tensors with dummy data
	dummyData1 := make([]byte, 1000*512*2) // F16 = 2 bytes per element
	dummyData2 := make([]byte, 512*1000*2) // F16 = 2 bytes per element

	tensors := []*ggml.Tensor{
		{
			Name:     "token_embd.weight",
			Kind:     1, // F16
			Shape:    []uint64{1000, 512},
			WriterTo: bytes.NewReader(dummyData1),
		},
		{
			Name:     "output.weight",
			Kind:     1, // F16
			Shape:    []uint64{512, 1000},
			WriterTo: bytes.NewReader(dummyData2),
		},
	}

	if err := ggml.WriteGGUF(file, kv, tensors); err != nil {
		file.Close()
		b.Fatal(err)
	}
	file.Close()

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
		_ = f.KeyValue("general.alignment").Uint()
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
