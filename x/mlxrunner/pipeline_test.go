package mlxrunner

import "testing"

func TestGenerationBudget(t *testing.T) {
	tests := []struct {
		name          string
		numPredict    int
		numCtx        int
		contextLength int
		promptLen     int
		want          int
	}{
		{
			// A checkpoint's max_position_embeddings is not a usable budget:
			// the request window is.
			name:          "open ended bounded by context window",
			numPredict:    -1,
			numCtx:        4096,
			contextLength: 131072,
			promptLen:     12,
			want:          40960,
		},
		{
			name:          "open ended bounded by model context when window is larger",
			numPredict:    -1,
			numCtx:        131072,
			contextLength: 8192,
			promptLen:     192,
			want:          8000,
		},
		{
			name:          "open ended without a context window uses the model context",
			numPredict:    -1,
			numCtx:        0,
			contextLength: 4096,
			promptLen:     96,
			want:          4000,
		},
		{
			name:          "explicit budget preserved",
			numPredict:    128,
			numCtx:        4096,
			contextLength: 131072,
			promptLen:     12,
			want:          128,
		},
		{
			name:          "explicit budget capped by model context",
			numPredict:    9000,
			numCtx:        131072,
			contextLength: 8192,
			promptLen:     192,
			want:          8000,
		},
		{
			name:          "zero budget still generates within the model context",
			numPredict:    0,
			numCtx:        4096,
			contextLength: 131072,
			promptLen:     12,
			want:          131060,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := generationBudget(tt.numPredict, tt.numCtx, tt.contextLength, tt.promptLen); got != tt.want {
				t.Fatalf("generationBudget(%d, %d, %d, %d) = %d, want %d",
					tt.numPredict, tt.numCtx, tt.contextLength, tt.promptLen, got, tt.want)
			}
		})
	}
}
