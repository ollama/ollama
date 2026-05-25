package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"slices"
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
			DataOffsets: [2]int{offset, offset + len(info.data)},
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
}

func TestCanonicalizeFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")

	// Hand-build a safetensors file where the data section is in
	// "scale, weight" order — the order MLX-C's unordered_map sometimes
	// produces — and tensorInfo fields are in declaration-default order.
	dataA := []byte{0xAA, 0xAA, 0xAA, 0xAA} // weight: 4 bytes
	dataB := []byte{0xBB, 0xBB}             // scale: 2 bytes

	header := map[string]any{
		"__metadata__": map[string]string{"quant_type": "demo"},
		"weight": tensorInfo{
			DataOffsets: [2]int{2, 6}, // weight stored at end
			Dtype:       "U8",
			Shape:       []int32{4},
		},
		"weight.scale": tensorInfo{
			DataOffsets: [2]int{0, 2}, // scale stored at start
			Dtype:       "U8",
			Shape:       []int32{2},
		},
	}
	headerJSON, _ := json.Marshal(header)
	padding := (8 - len(headerJSON)%8) % 8
	headerJSON = append(headerJSON, bytes.Repeat([]byte(" "), padding)...)

	f, _ := os.Create(path)
	binary.Write(f, binary.LittleEndian, uint64(len(headerJSON)))
	f.Write(headerJSON)
	f.Write(dataB) // scale first
	f.Write(dataA) // weight second
	f.Close()

	if err := CanonicalizeFile(path); err != nil {
		t.Fatalf("CanonicalizeFile: %v", err)
	}

	// Re-open and verify: tensors now appear in sorted name order, data
	// section ordered the same way, metadata preserved.
	ext, err := OpenForExtraction(path)
	if err != nil {
		t.Fatalf("OpenForExtraction: %v", err)
	}
	defer ext.Close()

	weight, err := ext.GetTensor("weight")
	if err != nil {
		t.Fatalf("GetTensor weight: %v", err)
	}
	scale, err := ext.GetTensor("weight.scale")
	if err != nil {
		t.Fatalf("GetTensor weight.scale: %v", err)
	}

	wb, err := io.ReadAll(weight.reader)
	if err != nil {
		t.Fatal(err)
	}
	sb, err := io.ReadAll(scale.reader)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(wb, dataA) {
		t.Errorf("weight bytes = %x, want %x", wb, dataA)
	}
	if !bytes.Equal(sb, dataB) {
		t.Errorf("scale bytes = %x, want %x", sb, dataB)
	}

	// "weight" sorts before "weight.scale", so weight should be at offset 0
	// in the data section after canonicalization.
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	hsz := binary.LittleEndian.Uint64(raw[:8])
	dataStart := 8 + int(hsz)
	if !bytes.Equal(raw[dataStart:dataStart+4], dataA) {
		t.Errorf("expected weight at offset 0, got %x", raw[dataStart:dataStart+4])
	}
}

// TestCanonicalizeFile_Idempotent confirms repeated canonicalization yields
// identical bytes — i.e., the on-disk format is the deterministic fixed point.
func TestCanonicalizeFile_Idempotent(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")
	a, b := []byte{1, 2, 3, 4}, []byte{5, 6, 7, 8}
	td1 := NewTensorDataFromBytes("a", "U8", []int32{4}, a)
	td2 := NewTensorDataFromBytes("b", "U8", []int32{4}, b)
	out, _ := io.ReadAll(BuildPackedSafetensorsReaderWithMetadata(
		[]*TensorData{td1, td2},
		map[string]string{"k": "v"},
	))
	os.WriteFile(path, out, 0o644)

	if err := CanonicalizeFile(path); err != nil {
		t.Fatalf("first canonicalize: %v", err)
	}
	first, _ := os.ReadFile(path)
	if err := CanonicalizeFile(path); err != nil {
		t.Fatalf("second canonicalize: %v", err)
	}
	second, _ := os.ReadFile(path)
	if !bytes.Equal(first, second) {
		t.Error("canonicalization is not idempotent")
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
			DataOffsets: [2]int{0, 8},
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
