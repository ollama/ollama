package cmd

import (
	"testing"
)

func TestValidateServerURL(t *testing.T) {
	tests := []struct {
		url     string
		wantErr bool
	}{
		{"http://localhost:11434", false},
		{"https://example.com", false},
		{"http://192.168.1.100:8080", false},
		{"https://ollama.example.com:443", false},
		{"ftp://example.com", true},
		{"invalid-url", true},
		{"http://", true},
		{"", true},
		{"://example.com", true},
	}

	for _, tt := range tests {
		t.Run(tt.url, func(t *testing.T) {
			err := validateServerURL(tt.url)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateServerURL(%q) error = %v, wantErr %v", tt.url, err, tt.wantErr)
			}
		})
	}
}