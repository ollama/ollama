package server

import (
	"bytes"
	"cmp"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestConvertFromSafetensorsPathValidation(t *testing.T) {
	// Store test data in the temp directory
	t.Setenv("OLLAMA_MODELS", cmp.Or(os.Getenv("OLLAMA_MODELS"), t.TempDir()))

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
		// Invalid test cases
		{
			name:     "InvalidRelativePathShallow",
			filePath: "../file.safetensors",
			wantErr:  errFilePath,
		},
		{
			name:     "InvalidRelativePathDeep",
			filePath: "../../../../../../data/file.txt",
			wantErr:  errFilePath,
		},
		{
			name:     "InvalidNestedPath",
			filePath: "dir/../../../../../other.safetensors",
			wantErr:  errFilePath,
		},

		// Valid relative path test cases
		{
			name:     "ValidRelativePath",
			filePath: "model.safetensors",
			wantErr:  nil,
		},
		{
			name:     "ValidRelativeWithDot",
			filePath: "./model.safetensors",
			wantErr:  nil,
		},
		{
			name:     "ValidNestedWithValidParentRef",
			filePath: "nested/../model.safetensors", // References tempDir/valid.safetensors
			wantErr:  nil,
		},

		// Absolute paths are rejected
		{
			name:     "AbsolutePathOutsideRoot",
			filePath: filepath.Join(os.TempDir(), "model.safetensors"),
			wantErr:  errFilePath, // Should fail since it's outside tmpDir
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
