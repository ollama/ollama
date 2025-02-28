package server

import (
	"bytes"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestConvertFromSafetensors(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	// Helper function to create a new layer and return its digest
	makeTemp := func(content string) string {
		l, err := NewLayer(strings.NewReader(content), "application/octet-stream")
		if err != nil {
			t.Fatalf("Failed to create layer: %v", err)
		}
		return l.Digest
	}

	// Create a safetensors compatible file with empty JSON content
	var buf bytes.Buffer
	headerSize := int64(len("{}"))
	binary.Write(&buf, binary.LittleEndian, headerSize)
	buf.WriteString("{}")

	model := makeTemp(buf.String())
	config := makeTemp(`{
		"architectures": ["LlamaForCausalLM"], 
		"vocab_size": 32000
	}`)
	tokenizer := makeTemp(`{
		"version": "1.0",
		"truncation": null,
		"padding": null,
		"added_tokens": [
			{
				"id": 0,
				"content": "<|endoftext|>",
				"single_word": false,
				"lstrip": false,
				"rstrip": false,
				"normalized": false,
				"special": true
			}
		]
	}`)

	tests := []struct {
		name     string
		filePath string
		wantErr  error
	}{
		// Invalid
		{
			name:     "InvalidRelativePathShallow",
			filePath: filepath.Join("..", "file.safetensors"),
			wantErr:  errFilePath,
		},
		{
			name:     "InvalidRelativePathDeep",
			filePath: filepath.Join("..", "..", "..", "..", "..", "..", "data", "file.txt"),
			wantErr:  errFilePath,
		},
		{
			name:     "InvalidNestedPath",
			filePath: filepath.Join("dir", "..", "..", "..", "..", "..", "other.safetensors"),
			wantErr:  errFilePath,
		},
		{
			name:     "AbsolutePathOutsideRoot",
			filePath: filepath.Join(os.TempDir(), "model.safetensors"),
			wantErr:  errFilePath, // Should fail since it's outside tmpDir
		},
		{
			name:     "ValidRelativePath",
			filePath: "model.safetensors",
			wantErr:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create the minimum required file map for convertFromSafetensors
			files := map[string]string{
				tt.filePath:      model,
				"config.json":    config,
				"tokenizer.json": tokenizer,
			}

			_, err := convertFromSafetensors(files, nil, false, func(resp api.ProgressResponse) {})

			if (tt.wantErr == nil && err != nil) ||
				(tt.wantErr != nil && err == nil) ||
				(tt.wantErr != nil && !errors.Is(err, tt.wantErr)) {
				t.Errorf("convertFromSafetensors() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
