package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
)

// tensorInfo holds tensor metadata from safetensors headers.
// This avoids depending on safetensors.go which requires the mlx tag.
type tensorInfo struct {
	Dtype       string  `json:"dtype"`
	Shape       []int32 `json:"shape"`
	DataOffsets [2]int  `json:"data_offsets"`
}

// TensorExtractor extracts individual tensors from a safetensors file.
// It provides io.Reader interfaces for each tensor's raw data, enabling
// streaming writes to blobs without loading entire tensors into memory.
type TensorExtractor struct {
	file       *os.File
	dataOffset int64 // Start of tensor data region
	header     map[string]tensorInfo
}

// TensorData holds tensor metadata and a reader for its raw bytes.
type TensorData struct {
	Name   string
	Dtype  string
	Shape  []int32
	Size   int64
	reader *io.SectionReader
}

// Reader returns an io.Reader for the tensor's raw bytes.
func (td *TensorData) Reader() io.Reader {
	return td.reader
}

// SafetensorsReader returns a reader that outputs the tensor wrapped in
// minimal safetensors format. This allows using mlx_load_safetensors on
// individual tensor blobs for native zero-copy loading.
func (td *TensorData) SafetensorsReader() io.Reader {
	// Build minimal safetensors header with tensor named "data"
	header := map[string]tensorInfo{
		"data": {
			Dtype:       td.Dtype,
			Shape:       td.Shape,
			DataOffsets: [2]int{0, int(td.Size)},
		},
	}
	headerJSON, _ := json.Marshal(header)

	// Pad header to 8-byte alignment
	padding := (8 - len(headerJSON)%8) % 8
	headerJSON = append(headerJSON, bytes.Repeat([]byte(" "), padding)...)

	// Build header with size prefix
	headerBuf := new(bytes.Buffer)
	binary.Write(headerBuf, binary.LittleEndian, uint64(len(headerJSON)))
	headerBuf.Write(headerJSON)

	// Return multi-reader: header + tensor data
	td.reader.Seek(0, io.SeekStart)
	return io.MultiReader(headerBuf, td.reader)
}

// SafetensorsSize returns the total size of the safetensors-wrapped tensor.
func (td *TensorData) SafetensorsSize() int64 {
	header := map[string]tensorInfo{
		"data": {
			Dtype:       td.Dtype,
			Shape:       td.Shape,
			DataOffsets: [2]int{0, int(td.Size)},
		},
	}
	headerJSON, _ := json.Marshal(header)
	padding := (8 - len(headerJSON)%8) % 8
	return 8 + int64(len(headerJSON)) + int64(padding) + td.Size
}

// OpenForExtraction opens a safetensors file for tensor extraction.
// The caller must call Close() when done.
func OpenForExtraction(path string) (*TensorExtractor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}

	var headerSize uint64
	if err := binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to read header size: %w", err)
	}

	headerBytes := make([]byte, headerSize)
	if _, err := f.Read(headerBytes); err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to read header: %w", err)
	}

	var header map[string]tensorInfo
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to parse header: %w", err)
	}

	delete(header, "__metadata__")

	return &TensorExtractor{
		file:       f,
		dataOffset: 8 + int64(headerSize), // 8 bytes for header size + header content
		header:     header,
	}, nil
}

// GetTensor returns tensor metadata and a reader for extracting a single tensor.
func (te *TensorExtractor) GetTensor(name string) (*TensorData, error) {
	info, ok := te.header[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q not found", name)
	}

	start := te.dataOffset + int64(info.DataOffsets[0])
	size := int64(info.DataOffsets[1] - info.DataOffsets[0])

	return &TensorData{
		Name:   name,
		Dtype:  info.Dtype,
		Shape:  info.Shape,
		Size:   size,
		reader: io.NewSectionReader(te.file, start, size),
	}, nil
}

// ListTensors returns all tensor names in sorted order.
func (te *TensorExtractor) ListTensors() []string {
	names := make([]string, 0, len(te.header))
	for name := range te.header {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// TensorCount returns the number of tensors in the file.
func (te *TensorExtractor) TensorCount() int {
	return len(te.header)
}

// Close closes the underlying file.
func (te *TensorExtractor) Close() error {
	return te.file.Close()
}

// ExtractAll returns TensorData for all tensors in the file.
// Each TensorData has a reader that reads from the original file.
// The caller must call Close() on the TensorExtractor when done.
func (te *TensorExtractor) ExtractAll() ([]*TensorData, error) {
	names := te.ListTensors()
	tensors := make([]*TensorData, 0, len(names))

	for _, name := range names {
		td, err := te.GetTensor(name)
		if err != nil {
			return nil, err
		}
		tensors = append(tensors, td)
	}

	return tensors, nil
}
