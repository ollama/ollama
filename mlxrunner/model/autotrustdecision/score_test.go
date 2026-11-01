package autotrustdecision

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/model/qwen3_5"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestBareDecisionPrompt(t *testing.T) {
	m := &Model{choiceCodes: []string{"A", "B", "C"}, choiceIDs: []int32{10, 11, 12}}
	m.head.Bias = make([]float32, 24)
	m.head.VerbalizerIDs = make([]int32, 24)
	got, prefix, ids, bias, err := m.prompt(llm.ScoreRequest{State: "Customer was charged twice."}, &llm.ScoreQuestion{
		Type: "choice", Instructions: "Which team should handle this?", Options: []string{"billing", "shipping", "support"},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := "[kind] choice\n[state] Customer was charged twice.\n[question] Which team should handle this?\n[options]\nA) billing\nB) shipping\nC) support\n[decision]:"
	if got != want || prefix != "[kind] choice\n[state] Customer was charged twice.\n[question] " {
		t.Fatalf("publisher bare-v1 prompt = %q, prefix = %q", got, prefix)
	}
	if !slices.Equal(ids, []int32{10, 11, 12}) || len(bias) != 3 {
		t.Fatalf("choice readout ids = %v, bias = %v", ids, bias)
	}
}

func TestImageCountLimit(t *testing.T) {
	m := &Model{}
	images := make([]api.ImageData, maxAutoTrustDecisionImages+1)
	_, _, _, _, err := m.prompt(llm.ScoreRequest{Images: images}, &llm.ScoreQuestion{Type: "noul"})
	if err == nil {
		t.Fatal("accepted more than the maximum image count")
	}
}

func TestChoiceCapacityUsesExistingAPILimit(t *testing.T) {
	const maxChoices = 255
	m := &Model{choiceCodes: make([]string, maxChoices), choiceIDs: make([]int32, maxChoices)}
	m.head.Bias = make([]float32, 24)
	for i := range m.choiceCodes {
		m.choiceCodes[i] = fmt.Sprintf("option-%d", i)
		m.choiceIDs[i] = int32(i)
	}
	q := &llm.ScoreQuestion{Type: "choice", Options: make([]string, maxChoices)}
	_, _, ids, _, err := m.prompt(llm.ScoreRequest{}, q)
	if err != nil || len(ids) != maxChoices {
		t.Fatalf("255 options: ids = %d, err = %v", len(ids), err)
	}
	q.Options = append(q.Options, "extra")
	if _, _, _, _, err := m.prompt(llm.ScoreRequest{}, q); err == nil {
		t.Fatal("accepted 256 options")
	}
}

func TestFinishScoreUsesLastPositionBiasAndTemperature(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			weight := mlx.FromValues([]float32{
				1, 0, 0, // token 0
				0, 1, 0, // token 1
				0, 0, 1, // token 2
				1, 1, 1, // token 3
			}, 4, 3)
			m := &Model{
				Model:       &qwen3_5.Model{LMHead: nn.NewLinear(weight, nil)},
				temperature: map[string]float32{"choice": 2},
			}
			hidden := mlx.FromValues([]float32{99, 99, 99, 2, 3, 5}, 1, 2, 3)
			got := m.finishScore(hidden, 2, []int32{3, 1, 0}, []float32{2, -1, 4}, "choice")
			want := []float32{6, 1, 3}
			if len(got) != len(want) {
				t.Fatalf("score count = %d, want %d", len(got), len(want))
			}
			for i, value := range got {
				if math.Abs(float64(value-want[i])) > 1e-6 {
					t.Fatalf("candidate %d: score = %g, want %g", i, value, want[i])
				}
			}
		})
	})
}
