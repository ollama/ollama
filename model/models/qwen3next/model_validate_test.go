package qwen3next

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/ml/nn"
)

func TestValidateRecurrentLayerRequiresSSMDT(t *testing.T) {
	m := &Model{
		Layers: []Layer{{
			Operator: &GatedDeltaNet{
				SSMQKV:     &nn.Linear{},
				SSMQKVGate: &nn.Linear{},
				SSMBeta:    &nn.Linear{},
				SSMAlpha:   &nn.Linear{},
			},
		}},
		Options: &Options{
			isRecurrent: []bool{true},
		},
	}

	err := m.Validate()
	if err == nil {
		t.Fatal("Validate() expected error, got nil")
	}
	if !strings.Contains(err.Error(), "missing ssm_dt") {
		t.Fatalf("unexpected error = %v", err)
	}
}

func TestValidateNonRecurrentSkipsLinearChecks(t *testing.T) {
	m := &Model{
		Layers: []Layer{{Operator: &FullAttention{}}},
		Options: &Options{
			isRecurrent: []bool{false},
		},
	}

	if err := m.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
}
