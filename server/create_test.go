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

func TestConvertFromSafetensors(t *testing.T) {
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
		// Invalid
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

func TestValidRelative(t *testing.T) {
	// Test cases organized as described in the discussion comment
	validPaths := []string{
		"x/y/z",        // nested path
		"a/b",          // simple path with one directory
		"o/l/l/a/m/a",  // multiple nested directories
		"file.txt",     // file in root
		"dir/file.txt", // file in subdirectory
		"a/b/",         // trailing slash
		"a//b",         // double slash
	}

	invalidPaths := []string{
		"/y",           // absolute path
		"/etc/passwd",  // absolute path
		"//etc/passwd", // double leading slash

		"./x/y",     // current directory reference
		"../x",      // parent directory reference
		"a/../../b", // traversal beyond root
		"a/../b",    // traversal within boundaries
		"a/./b",     // current directory in middle
		".",         // current directory only
		"..",        // parent directory only
		"a/b/..",    // ending with parent reference
		"a/b/.",     // ending with current reference
		"",          // empty path
	}

	// Test valid paths
	for _, path := range validPaths {
		t.Run("Valid: "+path, func(t *testing.T) {
			err := validRelative(path)
			if err != nil {
				t.Errorf("validRelative(%q) returned error %v, expected nil", path, err)
			}
		})
	}

	// Test invalid paths
	for _, path := range invalidPaths {
		t.Run("Invalid: "+path, func(t *testing.T) {
			err := validRelative(path)
			if err == nil {
				t.Errorf("validRelative(%q) returned nil, expected error", path)
			}
			if !errors.Is(err, errFilePath) {
				t.Errorf("validRelative(%q) returned error type %T, expected to wrap %T",
					path, err, errFilePath)
			}
		})
	}
}
