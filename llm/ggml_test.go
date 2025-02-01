package llm

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
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

func TestDecodeGGML(t *testing.T) {
	tests := []struct {
		name        string
		input       []byte
		wantErr     bool
		errContains string
	}{
		{
			name:        "Invalid magic",
			input:       []byte{0x00, 0x00, 0x00, 0x00},
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := bytes.NewReader(tt.input)
			_, _, err := DecodeGGML(reader, 0)

			fmt.Println(err)

			assert.Equal(t, err != nil, tt.wantErr)
			if tt.wantErr && err != nil {
				assert.Contains(t, err.Error(), tt.errContains)
			}
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

func makeTestBytes(magic uint32) []byte {
	buf := make([]byte, 4)
	binary.LittleEndian.PutUint32(buf, magic)
	return buf
}
