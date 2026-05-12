package glm4moelite

import (
	"testing"

	"github.com/ollama/ollama/ml/nn"
)

func TestValidate(t *testing.T) {
	tests := []struct {
		name    string
		model   *Model
		wantErr bool
	}{
		{
			name: "valid model with KB and VB",
			model: &Model{
				Layers: []Layer{
					{Attention: &Attention{KB: &nn.Linear{}, VB: &nn.Linear{}}},
				},
			},
			wantErr: false,
		},
		{
			name: "missing KB",
			model: &Model{
				Layers: []Layer{
					{Attention: &Attention{VB: &nn.Linear{}}},
				},
			},
			wantErr: true,
		},
		{
			name: "missing VB",
			model: &Model{
				Layers: []Layer{
					{Attention: &Attention{KB: &nn.Linear{}}},
				},
			},
			wantErr: true,
		},
		{
			name: "missing both KB and VB",
			model: &Model{
				Layers: []Layer{
					{Attention: &Attention{}},
				},
			},
			wantErr: true,
		},
		{
			name: "nil Attention is ok",
			model: &Model{
				Layers: []Layer{
					{Attention: nil},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.model.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr && err != ErrOldModelFormat {
				t.Errorf("Validate() error = %v, want %v", err, ErrOldModelFormat)
			}
		})
	}
}
