package llm

import (
	"math"
	"testing"
)

func TestValidateEmbedding(t *testing.T) {
	tests := []struct {
		name    string
		input   []float32
		wantErr bool
	}{
		{
			name:    "valid embedding",
			input:   []float32{0.1, 0.2, 0.3},
			wantErr: false,
		},
		{
			name:    "empty embedding",
			input:   []float32{},
			wantErr: false,
		},
		{
			name:    "nil embedding",
			input:   nil,
			wantErr: false,
		},
		{
			name:    "NaN at start",
			input:   []float32{float32(math.NaN()), 0.2, 0.3},
			wantErr: true,
		},
		{
			name:    "NaN in middle",
			input:   []float32{0.1, float32(math.NaN()), 0.3},
			wantErr: true,
		},
		{
			name:    "positive Inf",
			input:   []float32{float32(math.Inf(1)), 0.2, 0.3},
			wantErr: true,
		},
		{
			name:    "negative Inf",
			input:   []float32{float32(math.Inf(-1)), 0.2, 0.3},
			wantErr: true,
		},
		{
			name:    "all NaN",
			input:   []float32{float32(math.NaN()), float32(math.NaN()), float32(math.NaN())},
			wantErr: true,
		},
		{
			name:    "zeros are valid",
			input:   []float32{0, 0, 0},
			wantErr: false,
		},
		{
			name:    "negative values are valid",
			input:   []float32{-0.1, -0.2, -0.3},
			wantErr: false,
		},
		{
			name:    "MaxFloat32 is valid",
			input:   []float32{math.MaxFloat32, -math.MaxFloat32, 0.1},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateEmbedding(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateEmbedding() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
