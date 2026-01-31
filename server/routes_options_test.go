package server

import (
	"testing"
)

func TestModelOptionsNumCtxPriority(t *testing.T) {
	tests := []struct {
		name           string
		envContextLen  string // empty means not set (uses 0 sentinel)
		defaultNumCtx  int    // VRAM-based default
		modelNumCtx    int    // 0 means not set in model
		requestNumCtx  int    // 0 means not set in request
		expectedNumCtx int
	}{
		{
			name:           "vram default when nothing else set",
			envContextLen:  "",
			defaultNumCtx:  32768,
			modelNumCtx:    0,
			requestNumCtx:  0,
			expectedNumCtx: 32768,
		},
		{
			name:           "env var overrides vram default",
			envContextLen:  "8192",
			defaultNumCtx:  32768,
			modelNumCtx:    0,
			requestNumCtx:  0,
			expectedNumCtx: 8192,
		},
		{
			name:           "model overrides vram default",
			envContextLen:  "",
			defaultNumCtx:  32768,
			modelNumCtx:    16384,
			requestNumCtx:  0,
			expectedNumCtx: 16384,
		},
		{
			name:           "model overrides env var",
			envContextLen:  "8192",
			defaultNumCtx:  32768,
			modelNumCtx:    16384,
			requestNumCtx:  0,
			expectedNumCtx: 16384,
		},
		{
			name:           "request overrides everything",
			envContextLen:  "8192",
			defaultNumCtx:  32768,
			modelNumCtx:    16384,
			requestNumCtx:  4096,
			expectedNumCtx: 4096,
		},
		{
			name:           "request overrides vram default",
			envContextLen:  "",
			defaultNumCtx:  32768,
			modelNumCtx:    0,
			requestNumCtx:  4096,
			expectedNumCtx: 4096,
		},
		{
			name:           "request overrides model",
			envContextLen:  "",
			defaultNumCtx:  32768,
			modelNumCtx:    16384,
			requestNumCtx:  4096,
			expectedNumCtx: 4096,
		},
		{
			name:           "low vram tier default",
			envContextLen:  "",
			defaultNumCtx:  4096,
			modelNumCtx:    0,
			requestNumCtx:  0,
			expectedNumCtx: 4096,
		},
		{
			name:           "high vram tier default",
			envContextLen:  "",
			defaultNumCtx:  262144,
			modelNumCtx:    0,
			requestNumCtx:  0,
			expectedNumCtx: 262144,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set or clear environment variable
			if tt.envContextLen != "" {
				t.Setenv("OLLAMA_CONTEXT_LENGTH", tt.envContextLen)
			}

			// Create server with VRAM-based default
			s := &Server{
				defaultNumCtx: tt.defaultNumCtx,
			}

			// Create model options (use float64 as FromMap expects JSON-style numbers)
			var modelOpts map[string]any
			if tt.modelNumCtx != 0 {
				modelOpts = map[string]any{"num_ctx": float64(tt.modelNumCtx)}
			}
			model := &Model{
				Options: modelOpts,
			}

			// Create request options (use float64 as FromMap expects JSON-style numbers)
			var requestOpts map[string]any
			if tt.requestNumCtx != 0 {
				requestOpts = map[string]any{"num_ctx": float64(tt.requestNumCtx)}
			}

			opts, err := s.modelOptions(model, requestOpts)
			if err != nil {
				t.Fatalf("modelOptions failed: %v", err)
			}

			if opts.NumCtx != tt.expectedNumCtx {
				t.Errorf("NumCtx = %d, want %d", opts.NumCtx, tt.expectedNumCtx)
			}
		})
	}
}
