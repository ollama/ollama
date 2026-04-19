package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
)

// MaxHeaderSize bounds safetensors JSON headers before allocation.
const MaxHeaderSize = 100 << 20

// tensorInfo holds tensor metadata from safetensors headers.
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

// WithName returns a shallow copy of TensorData with a different logical tensor
// name but the same underlying raw data reader.
func (td *TensorData) WithName(name string) *TensorData {
	if td == nil {
		return nil
	}
	shape := make([]int32, len(td.Shape))
	copy(shape, td.Shape)
	return &TensorData{
		Name:   name,
		Dtype:  td.Dtype,
		Shape:  shape,
		Size:   td.Size,
		reader: td.reader,
	}
}

// Reader returns an io.Reader for the tensor's raw bytes.
func (td *TensorData) Reader() io.Reader {
	return td.reader
}

// safetensorsHeader builds the JSON header for a minimal safetensors blob
// containing a single tensor keyed by its name.
func (td *TensorData) safetensorsHeader() []byte {
	header := map[string]any{
		td.Name: tensorInfo{
			Dtype:       td.Dtype,
			Shape:       td.Shape,
			DataOffsets: [2]int{0, int(td.Size)},
		},
	}
	headerJSON, _ := json.Marshal(header)

	// Pad header to 8-byte alignment
	padding := (8 - len(headerJSON)%8) % 8
	headerJSON = append(headerJSON, bytes.Repeat([]byte(" "), padding)...)
	return headerJSON
}

// SafetensorsReader returns a reader that outputs the tensor wrapped in
// minimal safetensors format. This allows using mlx_load_safetensors on
// individual tensor blobs for native zero-copy loading.
// The tensor is keyed by its name in the safetensors header.
func (td *TensorData) SafetensorsReader() io.Reader {
	headerJSON := td.safetensorsHeader()

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
	headerJSON := td.safetensorsHeader()
	return 8 + int64(len(headerJSON)) + td.Size
}

// NewTensorDataFromBytes creates a TensorData from raw tensor bytes.
// This is useful for constructing packed blobs from already-extracted data.
func NewTensorDataFromBytes(name, dtype string, shape []int32, rawData []byte) *TensorData {
	return &TensorData{
		Name:   name,
		Dtype:  dtype,
		Shape:  shape,
		Size:   int64(len(rawData)),
		reader: io.NewSectionReader(bytes.NewReader(rawData), 0, int64(len(rawData))),
	}
}

// ExtractRawFromSafetensors reads a safetensors-wrapped reader and extracts
// the raw tensor data bytes (stripping the header).
func ExtractRawFromSafetensors(r io.Reader) ([]byte, error) {
	// Read header size (8 bytes, little endian)
	var headerSize uint64
	if err := binary.Read(r, binary.LittleEndian, &headerSize); err != nil {
		return nil, fmt.Errorf("failed to read header size: %w", err)
	}
	if headerSize > MaxHeaderSize {
		return nil, fmt.Errorf("safetensors header too large: %d bytes", headerSize)
	}
	if headerSize > uint64(math.MaxInt64) {
		return nil, fmt.Errorf("safetensors header size %d overflows int64", headerSize)
	}

	// Skip header
	if _, err := io.CopyN(io.Discard, r, int64(headerSize)); err != nil {
		return nil, fmt.Errorf("failed to skip header: %w", err)
	}

	// Read remaining bytes (the raw tensor data)
	return io.ReadAll(r)
}

// BuildPackedSafetensorsReader builds a streaming io.Reader that outputs a valid
// safetensors file containing multiple tensors. Used for packing expert tensors
// into a single blob without loading all data into memory.
// Each TensorData must have been obtained from GetTensor.
func BuildPackedSafetensorsReader(tensors []*TensorData) io.Reader {
	return BuildPackedSafetensorsReaderWithMetadata(tensors, nil)
}

// BuildPackedSafetensorsReaderWithMetadata builds a streaming io.Reader that
// outputs a valid safetensors file containing multiple tensors and optional
// metadata.
func BuildPackedSafetensorsReaderWithMetadata(tensors []*TensorData, metadata map[string]string) io.Reader {
	// Build the header with sequential data offsets
	header := make(map[string]any, len(tensors)+1)
	var offset int
	for _, td := range tensors {
		header[td.Name] = tensorInfo{
			Dtype:       td.Dtype,
			Shape:       td.Shape,
			DataOffsets: [2]int{offset, offset + int(td.Size)},
		}
		offset += int(td.Size)
	}
	if len(metadata) > 0 {
		header["__metadata__"] = metadata
	}

	headerJSON, _ := json.Marshal(header)

	// Pad header to 8-byte alignment
	padding := (8 - len(headerJSON)%8) % 8
	headerJSON = append(headerJSON, bytes.Repeat([]byte(" "), padding)...)

	// Build header with size prefix
	headerBuf := new(bytes.Buffer)
	binary.Write(headerBuf, binary.LittleEndian, uint64(len(headerJSON)))
	headerBuf.Write(headerJSON)

	// Build multi-reader: header + all tensor data readers
	readers := make([]io.Reader, 0, 1+len(tensors))
	readers = append(readers, headerBuf)
	for _, td := range tensors {
		td.reader.Seek(0, io.SeekStart)
		readers = append(readers, td.reader)
	}

	return io.MultiReader(readers...)
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

	stat, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to stat file: %w", err)
	}
	dataSize, err := validateSafetensorsHeaderSize(headerSize, stat.Size())
	if err != nil {
		f.Close()
		return nil, err
	}

	headerBytes := make([]byte, headerSize)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to read header: %w", err)
	}

	var header map[string]tensorInfo
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to parse header: %w", err)
	}

	delete(header, "__metadata__")
	if err := validateSafetensorsHeader(header, dataSize); err != nil {
		f.Close()
		return nil, err
	}

	return &TensorExtractor{
		file:       f,
		dataOffset: 8 + int64(headerSize), // 8 bytes for header size + header content
		header:     header,
	}, nil
}

func validateSafetensorsHeaderSize(headerSize uint64, fileSize int64) (int64, error) {
	if fileSize < 8 {
		return 0, fmt.Errorf("safetensors file is too small: %d bytes", fileSize)
	}
	if headerSize > MaxHeaderSize {
		return 0, fmt.Errorf("safetensors header too large: %d bytes", headerSize)
	}
	if headerSize > uint64(fileSize-8) {
		return 0, fmt.Errorf("safetensors header size %d exceeds file size %d", headerSize, fileSize)
	}
	if headerSize > uint64(math.MaxInt64-8) {
		return 0, fmt.Errorf("safetensors header size %d overflows int64", headerSize)
	}
	return fileSize - 8 - int64(headerSize), nil
}

func validateSafetensorsHeader(header map[string]tensorInfo, dataSize int64) error {
	type tensorRange struct {
		name       string
		start, end int64
	}

	ranges := make([]tensorRange, 0, len(header))
	for name, info := range header {
		if name == "" {
			return fmt.Errorf("safetensors tensor name is empty")
		}
		start := int64(info.DataOffsets[0])
		end := int64(info.DataOffsets[1])
		if start < 0 || end < start {
			return fmt.Errorf("safetensors tensor %q has invalid data offsets %v", name, info.DataOffsets)
		}
		if end > dataSize {
			return fmt.Errorf("safetensors tensor %q data offset %d exceeds data size %d", name, end, dataSize)
		}

		want, err := safetensorsTensorByteSize(info.Dtype, info.Shape)
		if err != nil {
			return fmt.Errorf("safetensors tensor %q: %w", name, err)
		}
		if end-start != want {
			return fmt.Errorf("safetensors tensor %q data size %d does not match dtype/shape size %d", name, end-start, want)
		}

		ranges = append(ranges, tensorRange{name: name, start: start, end: end})
	}

	sort.Slice(ranges, func(i, j int) bool {
		return ranges[i].start < ranges[j].start
	})
	for i := 1; i < len(ranges); i++ {
		if ranges[i].start < ranges[i-1].end {
			return fmt.Errorf("safetensors tensor %q overlaps tensor %q", ranges[i].name, ranges[i-1].name)
		}
	}
	return nil
}

func safetensorsTensorByteSize(dtype string, shape []int32) (int64, error) {
	elemSize, ok := safetensorsDTypeSize(dtype)
	if !ok {
		return 0, fmt.Errorf("unsupported dtype %q", dtype)
	}

	elements := int64(1)
	for _, dim := range shape {
		if dim < 0 {
			return 0, fmt.Errorf("invalid negative shape dimension %d", dim)
		}
		if dim != 0 && elements > math.MaxInt64/int64(dim) {
			return 0, fmt.Errorf("shape %v overflows int64", shape)
		}
		elements *= int64(dim)
	}
	if elements > math.MaxInt64/int64(elemSize) {
		return 0, fmt.Errorf("shape %v overflows int64", shape)
	}
	return elements * int64(elemSize), nil
}

func safetensorsDTypeSize(dtype string) (int, bool) {
	switch dtype {
	case "BOOL", "U8", "I8", "F8_E5M2", "F8_E5M2FNUZ", "F8_E4M3", "F8_E4M3FN", "F8_E4M3FNUZ":
		return 1, true
	case "U16", "I16", "F16", "BF16":
		return 2, true
	case "U32", "I32", "F32":
		return 4, true
	case "U64", "I64", "F64":
		return 8, true
	default:
		return 0, false
	}
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

// TensorMeta holds the metadata for a tensor in a safetensors blob.
type TensorMeta struct {
	Name  string
	Dtype string
	Shape []int32
}

// ReadBlobMetas reads metadata for every tensor in a safetensors blob file.
func ReadBlobMetas(path string) ([]TensorMeta, error) {
	ext, err := OpenForExtraction(path)
	if err != nil {
		return nil, err
	}
	defer ext.Close()

	names := ext.ListTensors()
	if len(names) == 0 {
		return nil, fmt.Errorf("no tensors found in %s", path)
	}

	metas := make([]TensorMeta, 0, len(names))
	for _, name := range names {
		info := ext.header[name]
		metas = append(metas, TensorMeta{
			Name:  name,
			Dtype: info.Dtype,
			Shape: info.Shape,
		})
	}

	return metas, nil
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
