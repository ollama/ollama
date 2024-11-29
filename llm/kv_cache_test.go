package llm

import (
	"testing"
)

func TestKvCacheBytesPerElement(t *testing.T) {
	tests := []struct {
		name      string
		cacheType string
		want      float64
	}{
		// Basic validation cases
		{name: "empty cache type", cacheType: "", want: 2.0},    // defaults to f16
		{name: "f16 standard", cacheType: "f16", want: 2.0},     // standard f16
		{name: "q8_0 standard", cacheType: "q8_0", want: 1.0},   // half of f16
		{name: "q4_0 standard", cacheType: "q4_0", want: 0.5},   // quarter of f16
		{name: "invalid type", cacheType: "invalid", want: 2.0}, // defaults to f16
		{name: "fp16 alias", cacheType: "fp16", want: 2.0},      // alias for f16
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := kvCacheBytesPerElement(tt.cacheType)
			if got != tt.want {
				t.Errorf("kvCacheBytesPerElement() = %v, want %v", got, tt.want)
			}
		})
	}
}
