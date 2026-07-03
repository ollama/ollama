package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// createTestSafetensors creates a minimal valid safetensors file with the given tensors.
func createTestSafetensors(t *testing.T, path string, tensors map[string]struct {
	dtype string
	shape []int32
	data  []byte
},
) {
	t.Helper()

	header := make(map[string]tensorInfo)
	var offset int
	var allData []byte

	// Sort names for deterministic file layout
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	slices.Sort(names)

	for _, name := range names {
		info := tensors[name]
		header[name] = tensorInfo{
			Dtype:       info.dtype,
			Shape:       info.shape,
			DataOffsets: []int64{int64(offset), int64(offset + len(info.data))},
		}
		allData = append(allData, info.data...)
		offset += len(info.data)
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("failed to marshal header: %v", err)
	}

	// Pad to 8-byte alignment
	padding := (8 - len(headerJSON)%8) % 8
	headerJSON = append(headerJSON, bytes.Repeat([]byte(" "), padding)...)

	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("failed to create file: %v", err)
	}
	defer f.Close()

	if err := binary.Write(f, binary.LittleEndian, uint64(len(headerJSON))); err != nil {
		t.Fatalf("failed to write header size: %v", err)
	}
	if _, err := f.Write(headerJSON); err != nil {
		t.Fatalf("failed to write header: %v", err)
	}
	if _, err := f.Write(allData); err != nil {
		t.Fatalf("failed to write data: %v", err)
	}
}

func createRawSafetensors(t *testing.T, path string, header map[string]any, data []byte) {
	t.Helper()

	headerJSON, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("failed to marshal header: %v", err)
	}
	padding := (8 - len(headerJSON)%8) % 8
	headerJSON = append(headerJSON, bytes.Repeat([]byte(" "), padding)...)

	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("failed to create file: %v", err)
	}
	defer f.Close()
	if err := binary.Write(f, binary.LittleEndian, uint64(len(headerJSON))); err != nil {
		t.Fatalf("failed to write header size: %v", err)
	}
	if _, err := f.Write(headerJSON); err != nil {
		t.Fatalf("failed to write header: %v", err)
	}
	if _, err := f.Write(data); err != nil {
		t.Fatalf("failed to write data: %v", err)
	}
}

func TestOpenForExtraction(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")

	// 4 float32 values = 16 bytes
	data := make([]byte, 16)
	binary.LittleEndian.PutUint32(data[0:4], 0x3f800000)   // 1.0
	binary.LittleEndian.PutUint32(data[4:8], 0x40000000)   // 2.0
	binary.LittleEndian.PutUint32(data[8:12], 0x40400000)  // 3.0
	binary.LittleEndian.PutUint32(data[12:16], 0x40800000) // 4.0

	createTestSafetensors(t, path, map[string]struct {
		dtype string
		shape []int32
		data  []byte
	}{
		"test_tensor": {dtype: "F32", shape: []int32{2, 2}, data: data},
	})

	ext, err := OpenForExtraction(path)
	if err != nil {
		t.Fatalf("OpenForExtraction failed: %v", err)
	}
	defer ext.Close()

	if ext.TensorCount() != 1 {
		t.Errorf("TensorCount() = %d, want 1", ext.TensorCount())
	}

	names := ext.ListTensors()
	if len(names) != 1 || names[0] != "test_tensor" {
		t.Errorf("ListTensors() = %v, want [test_tensor]", names)
	}
}

func TestOpenForExtractionRejectsInvalidHeaderSize(t *testing.T) {
	t.Run("header exceeds cap", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "test.safetensors")
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := binary.Write(f, binary.LittleEndian, uint64(maxSafetensorsHeaderSize+1)); err != nil {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}

		_, err = OpenForExtraction(path)
		if err == nil || !strings.Contains(err.Error(), "exceeds maximum") {
			t.Fatalf("OpenForExtraction() error = %v, want header size cap error", err)
		}
	})

	t.Run("header exceeds file", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "test.safetensors")
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := binary.Write(f, binary.LittleEndian, uint64(16)); err != nil {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}

		_, err = OpenForExtraction(path)
		if err == nil || !strings.Contains(err.Error(), "exceeds file payload") {
			t.Fatalf("OpenForExtraction() error = %v, want header/file size error", err)
		}
	})
}

func TestOpenForExtractionRejectsInvalidTensorMetadata(t *testing.T) {
	tests := []struct {
		name   string
		header map[string]any
		data   []byte
		want   string
	}{
		{
			name: "missing offsets",
			header: map[string]any{"weight": map[string]any{
				"dtype": "F32",
				"shape": []int32{1},
			}},
			data: make([]byte, 4),
			want: "invalid data offsets",
		},
		{
			name: "negative offset",
			header: map[string]any{"weight": tensorInfo{
				Dtype: "F32", Shape: []int32{1}, DataOffsets: []int64{-1, 4},
			}},
			data: make([]byte, 4),
			want: "negative data offsets",
		},
		{
			name: "reversed offsets",
			header: map[string]any{"weight": tensorInfo{
				Dtype: "F32", Shape: []int32{1}, DataOffsets: []int64{4, 0},
			}},
			data: make([]byte, 4),
			want: "invalid data offsets",
		},
		{
			name: "offset beyond file",
			header: map[string]any{"weight": tensorInfo{
				Dtype: "F32", Shape: []int32{2}, DataOffsets: []int64{0, 8},
			}},
			data: make([]byte, 4),
			want: "exceed data size",
		},
		{
			name: "unsupported dtype",
			header: map[string]any{"weight": tensorInfo{
				Dtype: "BAD", Shape: []int32{4}, DataOffsets: []int64{0, 4},
			}},
			data: make([]byte, 4),
			want: "unsupported dtype",
		},
		{
			name: "negative shape",
			header: map[string]any{"weight": tensorInfo{
				Dtype: "F32", Shape: []int32{-1}, DataOffsets: []int64{0, 0},
			}},
			data: nil,
			want: "negative dimension",
		},
		{
			name: "shape byte mismatch",
			header: map[string]any{"weight": tensorInfo{
				Dtype: "F32", Shape: []int32{2}, DataOffsets: []int64{0, 4},
			}},
			data: make([]byte, 4),
			want: "does not match",
		},
		{
			name: "overlapping offsets",
			header: map[string]any{
				"a": tensorInfo{Dtype: "F32", Shape: []int32{1}, DataOffsets: []int64{0, 4}},
				"b": tensorInfo{Dtype: "F32", Shape: []int32{1}, DataOffsets: []int64{2, 6}},
			},
			data: make([]byte, 6),
			want: "overlap",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "test.safetensors")
			createRawSafetensors(t, path, tt.header, tt.data)
			_, err := OpenForExtraction(path)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("OpenForExtraction() error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestGetTensor(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")

	data := make([]byte, 16)
	for i := range 4 {
		binary.LittleEndian.PutUint32(data[i*4:], uint32(i+1))
	}

	createTestSafetensors(t, path, map[string]struct {
		dtype string
		shape []int32
		data  []byte
	}{
		"weight": {dtype: "F32", shape: []int32{2, 2}, data: data},
	})

	ext, err := OpenForExtraction(path)
	if err != nil {
		t.Fatalf("OpenForExtraction failed: %v", err)
	}
	defer ext.Close()

	td, err := ext.GetTensor("weight")
	if err != nil {
		t.Fatalf("GetTensor failed: %v", err)
	}

	if td.Name != "weight" {
		t.Errorf("Name = %q, want %q", td.Name, "weight")
	}
	if td.Dtype != "F32" {
		t.Errorf("Dtype = %q, want %q", td.Dtype, "F32")
	}
	if td.Size != 16 {
		t.Errorf("Size = %d, want 16", td.Size)
	}
	if len(td.Shape) != 2 || td.Shape[0] != 2 || td.Shape[1] != 2 {
		t.Errorf("Shape = %v, want [2 2]", td.Shape)
	}

	// Read the raw data
	rawData, err := io.ReadAll(td.Reader())
	if err != nil {
		t.Fatalf("Reader() read failed: %v", err)
	}
	if len(rawData) != 16 {
		t.Errorf("raw data length = %d, want 16", len(rawData))
	}
}

func TestGetTensor_NotFound(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")

	createTestSafetensors(t, path, map[string]struct {
		dtype string
		shape []int32
		data  []byte
	}{
		"exists": {dtype: "F32", shape: []int32{1}, data: make([]byte, 4)},
	})

	ext, err := OpenForExtraction(path)
	if err != nil {
		t.Fatalf("OpenForExtraction failed: %v", err)
	}
	defer ext.Close()

	_, err = ext.GetTensor("missing")
	if err == nil {
		t.Error("expected error for missing tensor, got nil")
	}
}

func TestSafetensorsReaderRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")

	data := make([]byte, 16)
	for i := range 4 {
		binary.LittleEndian.PutUint32(data[i*4:], uint32(0x3f800000+i))
	}

	createTestSafetensors(t, path, map[string]struct {
		dtype string
		shape []int32
		data  []byte
	}{
		"tensor_a": {dtype: "F32", shape: []int32{2, 2}, data: data},
	})

	ext, err := OpenForExtraction(path)
	if err != nil {
		t.Fatalf("OpenForExtraction failed: %v", err)
	}
	defer ext.Close()

	td, err := ext.GetTensor("tensor_a")
	if err != nil {
		t.Fatalf("GetTensor failed: %v", err)
	}

	// Wrap as safetensors and extract back
	stReader := td.SafetensorsReader()
	stData, err := io.ReadAll(stReader)
	if err != nil {
		t.Fatalf("SafetensorsReader read failed: %v", err)
	}

	// Verify size
	if int64(len(stData)) != td.SafetensorsSize() {
		t.Errorf("SafetensorsSize() = %d, actual = %d", td.SafetensorsSize(), len(stData))
	}

	// Extract raw data back
	raw, err := ExtractRawFromSafetensors(bytes.NewReader(stData))
	if err != nil {
		t.Fatalf("ExtractRawFromSafetensors failed: %v", err)
	}

	if !bytes.Equal(raw, data) {
		t.Errorf("round-trip data mismatch: got %v, want %v", raw, data)
	}
}

func TestNewTensorDataFromBytes(t *testing.T) {
	data := []byte{1, 2, 3, 4}
	td := NewTensorDataFromBytes("test", "U8", []int32{4}, data)

	if td.Name != "test" {
		t.Errorf("Name = %q, want %q", td.Name, "test")
	}
	if td.Size != 4 {
		t.Errorf("Size = %d, want 4", td.Size)
	}

	rawData, err := io.ReadAll(td.Reader())
	if err != nil {
		t.Fatalf("Reader() failed: %v", err)
	}
	if !bytes.Equal(rawData, data) {
		t.Errorf("data mismatch: got %v, want %v", rawData, data)
	}
}

func TestBuildPackedSafetensorsReader(t *testing.T) {
	data1 := []byte{1, 2, 3, 4}
	data2 := []byte{5, 6, 7, 8, 9, 10, 11, 12}

	td1 := NewTensorDataFromBytes("a", "U8", []int32{4}, data1)
	td2 := NewTensorDataFromBytes("b", "U8", []int32{8}, data2)

	packed := BuildPackedSafetensorsReader([]*TensorData{td1, td2})
	packedBytes, err := io.ReadAll(packed)
	if err != nil {
		t.Fatalf("BuildPackedSafetensorsReader read failed: %v", err)
	}

	// Verify it's a valid safetensors file by parsing the header
	var headerSize uint64
	if err := binary.Read(bytes.NewReader(packedBytes), binary.LittleEndian, &headerSize); err != nil {
		t.Fatalf("failed to read header size: %v", err)
	}

	headerJSON := packedBytes[8 : 8+headerSize]
	var header map[string]tensorInfo
	if err := json.Unmarshal(headerJSON, &header); err != nil {
		t.Fatalf("failed to parse header: %v", err)
	}

	if len(header) != 2 {
		t.Errorf("header has %d entries, want 2", len(header))
	}

	infoA, ok := header["a"]
	if !ok {
		t.Fatal("tensor 'a' not found in header")
	}
	if infoA.Dtype != "U8" {
		t.Errorf("tensor 'a' dtype = %q, want %q", infoA.Dtype, "U8")
	}

	infoB, ok := header["b"]
	if !ok {
		t.Fatal("tensor 'b' not found in header")
	}

	// Verify data region contains both tensors
	dataStart := 8 + int(headerSize)
	dataRegion := packedBytes[dataStart:]
	if infoA.DataOffsets[0] == 0 {
		// a comes first
		if !bytes.Equal(dataRegion[:4], data1) {
			t.Error("tensor 'a' data mismatch")
		}
		if !bytes.Equal(dataRegion[infoB.DataOffsets[0]:infoB.DataOffsets[1]], data2) {
			t.Error("tensor 'b' data mismatch")
		}
	} else {
		// b comes first
		if !bytes.Equal(dataRegion[:8], data2) {
			t.Error("tensor 'b' data mismatch")
		}
	}
}

func TestExtractAll(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")

	createTestSafetensors(t, path, map[string]struct {
		dtype string
		shape []int32
		data  []byte
	}{
		"alpha": {dtype: "F32", shape: []int32{2}, data: make([]byte, 8)},
		"beta":  {dtype: "F16", shape: []int32{4}, data: make([]byte, 8)},
	})

	ext, err := OpenForExtraction(path)
	if err != nil {
		t.Fatalf("OpenForExtraction failed: %v", err)
	}
	defer ext.Close()

	tensors, err := ext.ExtractAll()
	if err != nil {
		t.Fatalf("ExtractAll failed: %v", err)
	}

	if len(tensors) != 2 {
		t.Errorf("ExtractAll returned %d tensors, want 2", len(tensors))
	}

	// Verify sorted order
	if tensors[0].Name != "alpha" || tensors[1].Name != "beta" {
		t.Errorf("tensors not in sorted order: %s, %s", tensors[0].Name, tensors[1].Name)
	}
}

func TestExtractRawFromSafetensors_InvalidInput(t *testing.T) {
	// Empty reader
	_, err := ExtractRawFromSafetensors(bytes.NewReader(nil))
	if err == nil {
		t.Error("expected error for empty reader")
	}

	// Truncated header size
	_, err = ExtractRawFromSafetensors(bytes.NewReader([]byte{1, 2, 3}))
	if err == nil {
		t.Error("expected error for truncated header size")
	}

	// Zero header size
	headerSizeBytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(headerSizeBytes, 0)
	_, err = ExtractRawFromSafetensors(bytes.NewReader(headerSizeBytes))
	if err == nil {
		t.Error("expected error for zero header size")
	}

	// Oversized header
	binary.LittleEndian.PutUint64(headerSizeBytes, uint64(maxSafetensorsHeaderSize)+1)
	_, err = ExtractRawFromSafetensors(bytes.NewReader(headerSizeBytes))
	if err == nil {
		t.Error("expected error for oversized header size")
	}
}

func TestOpenForExtraction_MetadataIgnored(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")

	// Manually create a safetensors file with __metadata__
	header := map[string]any{
		"__metadata__": map[string]string{"format": "pt"},
		"weight": tensorInfo{
			Dtype:       "F32",
			Shape:       []int32{2},
			DataOffsets: []int64{0, 8},
		},
	}
	headerJSON, _ := json.Marshal(header)
	padding := (8 - len(headerJSON)%8) % 8
	headerJSON = append(headerJSON, bytes.Repeat([]byte(" "), padding)...)

	f, _ := os.Create(path)
	binary.Write(f, binary.LittleEndian, uint64(len(headerJSON)))
	f.Write(headerJSON)
	f.Write(make([]byte, 8))
	f.Close()

	ext, err := OpenForExtraction(path)
	if err != nil {
		t.Fatalf("OpenForExtraction failed: %v", err)
	}
	defer ext.Close()

	// __metadata__ should be stripped
	if ext.TensorCount() != 1 {
		t.Errorf("TensorCount() = %d, want 1 (metadata should be stripped)", ext.TensorCount())
	}
}
