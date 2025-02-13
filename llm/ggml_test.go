package llm

import (
	"bytes"
	"encoding/binary"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDetectGGMLType(t *testing.T) {
	tests := []struct {
		name     string
		magic    uint32
		expected string
	}{
		{"GGML", FILE_MAGIC_GGML, "ggml"},
		{"GGMF", FILE_MAGIC_GGMF, "ggmf"},
		{"GGJT", FILE_MAGIC_GGJT, "ggjt"},
		{"GGLA", FILE_MAGIC_GGLA, "ggla"},
		{"GGUF_LE", FILE_MAGIC_GGUF_LE, "gguf"},
		{"GGUF_BE", FILE_MAGIC_GGUF_BE, "gguf"},
		{"Unknown", 0x12345678, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := make([]byte, 4)
			binary.LittleEndian.PutUint32(buf, tt.magic)
			got := DetectGGMLType(buf)
			assert.Equal(t, got, tt.expected)
		})
	}
}

func TestKV(t *testing.T) {
	kv := KV{
		"general.architecture":          "llama",
		"general.file_type":             uint32(1),
		"general.parameter_count":       uint64(7000000000),
		"llama.context_length":          uint32(4096),
		"llama.embedding_length":        uint32(4096),
		"llama.block_count":             uint32(32),
		"llama.attention.head_count":    uint32(32),
		"llama.attention.head_count_kv": uint32(8),
		"tokenizer.chat_template":       "test template",
	}

	tests := []struct {
		name     string
		fn       func() interface{}
		expected interface{}
	}{
		{"Architecture", func() interface{} { return kv.Architecture() }, "llama"},
		{"FileType", func() interface{} { return kv.FileType() }, fileType(1)},
		{"ParameterCount", func() interface{} { return kv.ParameterCount() }, uint64(7000000000)},
		{"ContextLength", func() interface{} { return kv.ContextLength() }, uint64(4096)},
		{"EmbeddingLength", func() interface{} { return kv.EmbeddingLength() }, uint64(4096)},
		{"BlockCount", func() interface{} { return kv.BlockCount() }, uint64(32)},
		{"HeadCount", func() interface{} { return kv.HeadCount() }, uint64(32)},
		{"HeadCountKV", func() interface{} { return kv.HeadCountKV() }, uint64(8)},
		{"GQA", func() interface{} { return kv.GQA() }, uint64(4)},
		{"ChatTemplate", func() interface{} { return kv.ChatTemplate() }, "test template"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.fn()
			assert.Equal(t, got, tt.expected)
		})
	}
}

func TestTensor(t *testing.T) {
	tests := []struct {
		name          string
		tensor        Tensor
		wantParams    uint64
		wantTypeSize  uint64
		wantBlockSize uint64
		wantSize      uint64
	}{
		{
			name: "F32 Tensor",
			tensor: Tensor{
				Name:   "test",
				Kind:   0, // F32
				Shape:  []uint64{2, 3, 4},
				Offset: 0,
			},
			wantParams:    24, // 2 * 3 * 4
			wantBlockSize: 1,
			wantTypeSize:  4,
			wantSize:      96, // params * typeSize / blockSize
		},
		{
			name: "Q4_0 Tensor",
			tensor: Tensor{
				Name:   "test",
				Kind:   2, // Q4_0
				Shape:  []uint64{32, 32},
				Offset: 0,
			},
			wantParams:    1024, // 32 * 32
			wantBlockSize: 32,
			wantTypeSize:  18,  // 2 + blockSize/2
			wantSize:      576, // params * typeSize / blockSize
		},
		{
			name: "Others Tensor",
			tensor: Tensor{
				Name:   "test",
				Kind:   11, // Q3_K
				Shape:  []uint64{32, 32},
				Offset: 0,
			},
			wantParams:    1024, // 32 * 32
			wantBlockSize: 256,
			wantTypeSize:  110, // blockSize/8 + blockSize/4 + 12 + 2
			wantSize:      440, // params * typeSize / blockSize
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.tensor.parameters(), tt.wantParams)
			assert.Equal(t, tt.tensor.typeSize(), tt.wantTypeSize)
			assert.Equal(t, tt.tensor.blockSize(), tt.wantBlockSize)
			assert.Equal(t, tt.tensor.Size(), tt.wantSize)
		})
	}
}

func TestDecodeGGML(t *testing.T) {
	tests := []struct {
		name        string
		input       []byte
		wantErr     bool
		errContains string
	}{
		{
			name:        "Invalid magic",
			input:       makeTestBytes(0x00000000),
			wantErr:     true,
			errContains: "invalid file magic",
		},
		{
			name:        "GGML format",
			input:       makeTestBytes(FILE_MAGIC_GGML),
			wantErr:     true,
			errContains: "unsupported model format",
		},
		{
			name:        "Empty input",
			input:       []byte{},
			wantErr:     true,
			errContains: "EOF",
		},
		{
			name: "GGLA invalid version",
			input: makeTestBytes(FILE_MAGIC_GGLA,
				withGGLAVersion(2)), // only version 1 is valid
			wantErr:     true,
			errContains: "invalid version",
		},
		{
			name: "GGLA incomplete header",
			input: makeTestBytes(FILE_MAGIC_GGLA,
				withGGLAVersion(1)), // missing r and alpha
			wantErr:     true,
			errContains: "EOF",
		},
		{
			name: "GGLA valid minimal",
			input: makeTestBytes(FILE_MAGIC_GGLA,
				withGGLAVersion(1),
				withGGLAHeader(32, 1)),
			wantErr: false,
		},
		{
			name: "GGLA with tensor",
			input: makeTestBytes(FILE_MAGIC_GGLA,
				withGGLAVersion(1),
				withGGLAHeader(32, 1),
				withGGLATensor(2, 4, 0, []uint32{2, 3}, "test")),
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := bytes.NewReader(tt.input)
			model, _, err := DecodeGGML(r, -1)
			if tt.wantErr {
				require.Error(t, err)
				require.Contains(t, err.Error(), tt.errContains)
				return
			}
			require.NoError(t, err)
			require.NotNil(t, model)
		})
	}
}

func TestSupportsKVCacheType(t *testing.T) {
	ggml := GGML{}

	tests := []struct {
		name      string
		cacheType string
		want      bool
	}{
		{"F16", "f16", true},
		{"Q8_0", "q8_0", true},
		{"Q4_0", "q4_0", true},
		{"Invalid", "invalid", false},
		{"Empty", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ggml.SupportsKVCacheType(tt.cacheType)
			assert.Equal(t, got, tt.want)
		})
	}
}

func TestKVCacheBytesPerElement(t *testing.T) {
	tests := []struct {
		name      string
		cacheType string
		want      float64
	}{
		{"F16", "f16", 2.0},
		{"Q8_0", "q8_0", 1.0},
		{"Q4_0", "q4_0", 0.5},
		{"Default", "invalid", 2.0},
		{"Empty", "", 2.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := kvCacheBytesPerElement(tt.cacheType)
			assert.Equal(t, got, tt.want)
		})
	}
}

func makeTestBytes(magic uint32, opts ...testBytesOption) []byte {
	buf := new(bytes.Buffer)
	binary.Write(buf, binary.LittleEndian, magic)
	for _, opt := range opts {
		opt(buf)
	}
	return buf.Bytes()
}

type testBytesOption func(*bytes.Buffer)

// GGLA options
func withGGLAVersion(version uint32) testBytesOption {
	return func(buf *bytes.Buffer) {
		binary.Write(buf, binary.LittleEndian, version)
	}
}

func withGGLAHeader(r, alpha uint32) testBytesOption {
	return func(buf *bytes.Buffer) {
		binary.Write(buf, binary.LittleEndian, r)
		binary.Write(buf, binary.LittleEndian, alpha)
	}
}

func withGGLATensor(dims uint32, namesize uint32, kind uint32, shape []uint32, name string) testBytesOption {
	return func(buf *bytes.Buffer) {
		binary.Write(buf, binary.LittleEndian, dims)
		binary.Write(buf, binary.LittleEndian, namesize)
		binary.Write(buf, binary.LittleEndian, kind)
		for _, s := range shape {
			binary.Write(buf, binary.LittleEndian, s)
		}
		buf.WriteString(name)

		// Pad to 32-byte boundary
		padding := make([]byte, (32-buf.Len()%32)%32)
		buf.Write(padding)

		// Write dummy tensor data
		var size uint32 = 1
		for _, s := range shape {
			size *= s
		}
		tensorData := make([]byte, size*4) // assuming float32 (4 bytes)
		buf.Write(tensorData)
	}
}
