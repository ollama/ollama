package server

import (
	"errors"
	"testing"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
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

func TestModelOptionsGenerationDefaultsPriority(t *testing.T) {
	m := &Model{
		GenerationDefaults: model.GenerationDefaults{
			"top_k":          int64(12),
			"top_p":          float64(0.7),
			"min_p":          float64(0.05),
			"temperature":    float64(0.4),
			"repeat_last_n":  int64(128),
			"repeat_penalty": float64(1.2),
		},
		Options: map[string]any{
			"top_p":         float64(0.5),
			"min_p":         float64(0),
			"repeat_last_n": float64(0),
		},
	}
	requestOpts := map[string]any{
		"temperature":    float64(0),
		"repeat_penalty": float64(1.5),
	}

	opts, err := (&Server{}).modelOptions(m, requestOpts)
	if err != nil {
		t.Fatal(err)
	}

	if opts.TopK != 12 {
		t.Fatalf("TopK = %d, want 12", opts.TopK)
	}
	if opts.TopP != 0.5 {
		t.Fatalf("TopP = %v, want 0.5", opts.TopP)
	}
	if opts.MinP != 0 {
		t.Fatalf("MinP = %v, want 0", opts.MinP)
	}
	if opts.Temperature != 0 {
		t.Fatalf("Temperature = %v, want 0", opts.Temperature)
	}
	if opts.RepeatLastN != 0 {
		t.Fatalf("RepeatLastN = %d, want 0", opts.RepeatLastN)
	}
	if opts.RepeatPenalty != 1.5 {
		t.Fatalf("RepeatPenalty = %v, want 1.5", opts.RepeatPenalty)
	}
}

func TestModelOptionsEmbeddingNumBatchDefault(t *testing.T) {
	tests := []struct {
		name             string
		defaultNumCtx    int
		capabilities     []string
		modelOpts        map[string]any
		requestOpts      map[string]any
		expectedNumBatch int
	}{
		{
			name:             "embedding model defaults to embedding batch size",
			defaultNumCtx:    40960,
			capabilities:     []string{string(model.CapabilityEmbedding)},
			expectedNumBatch: llm.DefaultEmbeddingNumBatch,
		},
		{
			name:             "embedding default is capped at launch, not by num_ctx",
			defaultNumCtx:    40960,
			capabilities:     []string{string(model.CapabilityEmbedding)},
			requestOpts:      map[string]any{"num_ctx": float64(30)},
			expectedNumBatch: llm.DefaultEmbeddingNumBatch,
		},
		{
			name:             "model num_batch overrides embedding default",
			defaultNumCtx:    40960,
			capabilities:     []string{string(model.CapabilityEmbedding)},
			modelOpts:        map[string]any{"num_batch": float64(1024)},
			expectedNumBatch: 1024,
		},
		{
			name:             "request num_batch overrides embedding default",
			defaultNumCtx:    40960,
			capabilities:     []string{string(model.CapabilityEmbedding)},
			requestOpts:      map[string]any{"num_batch": float64(4096)},
			expectedNumBatch: 4096,
		},
		{
			name:             "non embedding model keeps general default",
			defaultNumCtx:    40960,
			capabilities:     []string{string(model.CapabilityCompletion)},
			expectedNumBatch: 512,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &Server{defaultNumCtx: tt.defaultNumCtx}
			m := &Model{
				Options: tt.modelOpts,
			}
			m.Config.Capabilities = tt.capabilities

			opts, err := s.modelOptions(m, tt.requestOpts)
			if err != nil {
				t.Fatalf("modelOptions failed: %v", err)
			}

			if opts.NumBatch != tt.expectedNumBatch {
				t.Fatalf("NumBatch = %d, want %d", opts.NumBatch, tt.expectedNumBatch)
			}
		})
	}
}

func TestModelOptionsDraftNumPredictDefault(t *testing.T) {
	tests := []struct {
		name        string
		model       *Model
		requestOpts map[string]any
		want        int
	}{
		{
			name:  "separate draft model keeps default enabled",
			model: &Model{DraftPath: "draft.gguf"},
			want:  4,
		},
		{
			name:  "embedded draft requires explicit parameter",
			model: &Model{},
			want:  0,
		},
		{
			name:  "model parameter enables embedded draft",
			model: &Model{Options: map[string]any{"draft_num_predict": float64(4)}},
			want:  4,
		},
		{
			name:        "request parameter enables embedded draft",
			model:       &Model{},
			requestOpts: map[string]any{"draft_num_predict": float64(8)},
			want:        8,
		},
		{
			name:        "request can disable separate draft model",
			model:       &Model{DraftPath: "draft.gguf"},
			requestOpts: map[string]any{"draft_num_predict": float64(0)},
			want:        0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts, err := (&Server{}).modelOptions(tt.model, tt.requestOpts)
			if err != nil {
				t.Fatal(err)
			}
			if opts.DraftNumPredict != tt.want {
				t.Fatalf("DraftNumPredict = %d, want %d", opts.DraftNumPredict, tt.want)
			}
		})
	}
}

func TestUsesAutomaticNumBatch(t *testing.T) {
	tests := []struct {
		name        string
		modelOpts   map[string]any
		requestOpts map[string]any
		want        bool
	}{
		{
			name: "default is automatic",
			want: true,
		},
		{
			name:        "model num_batch is explicit",
			modelOpts:   map[string]any{"num_batch": float64(1024)},
			requestOpts: nil,
			want:        false,
		},
		{
			name:        "request num_batch is explicit",
			requestOpts: map[string]any{"num_batch": float64(2048)},
			want:        false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := usesAutomaticNumBatch(&Model{Options: tt.modelOpts}, tt.requestOpts); got != tt.want {
				t.Fatalf("usesAutomaticNumBatch = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestModelOptionsInvalidRequestOptionType(t *testing.T) {
	s := &Server{}
	model := &Model{}

	// A wrong-typed request option is marked as a client error.
	_, err := s.modelOptions(model, map[string]any{"temperature": "hot"})
	if err == nil {
		t.Fatal("expected error for wrong-typed temperature option")
	}
	var invalidOpt errInvalidOption
	if !errors.As(err, &invalidOpt) {
		t.Fatalf("expected errInvalidOption, got %T: %v", err, err)
	}

	// The same error from model options is a server-side problem, not marked.
	badModel := &Model{Options: map[string]any{"temperature": "hot"}}
	_, err = s.modelOptions(badModel, nil)
	if err == nil {
		t.Fatal("expected error for wrong-typed model option")
	}
	if errors.As(err, &invalidOpt) {
		t.Fatalf("model option error should not be errInvalidOption, got %T: %v", err, err)
	}
}

func TestScheduleErrorInvalidOption(t *testing.T) {
	inner := errors.New(`option "temperature" must be of type float32`)

	status, message := scheduleError("model", errInvalidOption{inner})
	if status != 400 {
		t.Fatalf("expected status 400, got %d", status)
	}
	if message != inner.Error() {
		t.Fatalf("expected message %q, got %q", inner.Error(), message)
	}

	status, _ = scheduleError("model", inner)
	if status != 500 {
		t.Fatalf("expected status 500 for unmarked error, got %d", status)
	}
}
