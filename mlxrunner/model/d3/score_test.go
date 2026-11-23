package d3

import (
	"math"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/model/qwen3_5"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestScorePrompt(t *testing.T) {
	m := &Model{config: config{Codes: []string{"A", "B"}}}
	input := llm.ScoreRequest{State: "A damaged order.", Images: []api.ImageData{{1}, {2}}}
	question := &llm.ScoreQuestion{Instructions: "Which team?", Options: []string{"returns: Refunds", "billing: Payments"}}
	prompt, bodyStart, err := m.scorePrompt(input, question)
	if err != nil {
		t.Fatal(err)
	}
	want := strings.Repeat(imageMarker, 2) + "State:\nA damaged order.\n\nQuestion:\nWhich team?\n\nOptions:\nA: returns: Refunds\nB: billing: Payments\n\nReply with only the code of the best option."
	if got := prompt[bodyStart : bodyStart+len(want)]; got != want {
		t.Fatalf("user prompt = %q, want %q", got, want)
	}
	if !strings.Contains(prompt[:bodyStart], systemPrompt) {
		t.Fatalf("system prompt missing from %q", prompt)
	}

	question.Options = nil
	if _, _, err := m.scorePrompt(input, question); err == nil {
		t.Fatal("accepted a question with no options")
	}
	question.Options = []string{"one", "two", "three"}
	if _, _, err := m.scorePrompt(input, question); err == nil {
		t.Fatal("accepted more options than the decision codebook")
	}
}

func TestNoncausalForwardAndReadout(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		identity := []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}
		linear := func(values []float32, rows int) *nn.Linear {
			return &nn.Linear{Weight: mlx.FromValues(values, rows, 4)}
		}
		norm := func() *nn.RMSNorm {
			return &nn.RMSNorm{Weight: mlx.FromValues([]float32{1, 1, 1, 1}, 4)}
		}
		qGate := append(append([]float32(nil), identity...), make([]float32, 16)...)
		zeroMLP := &qwen3_5.DenseMLP{
			GateProj: linear(make([]float32, 16), 4),
			UpProj:   linear(make([]float32, 16), 4),
			DownProj: linear(make([]float32, 16), 4),
		}
		layers := make([]*qwen3_5.Layer, 2)
		for i := range layers {
			layers[i] = &qwen3_5.Layer{
				InputNorm: norm(), PostAttentionNorm: norm(), MLP: zeroMLP,
				FullAttn: &qwen3_5.FullAttention{
					QProj: linear(qGate, 8), KProj: linear(identity, 4),
					VProj: linear(identity, 4), OProj: linear(identity, 4),
					QNorm: norm(), KNorm: norm(),
				},
			}
		}
		cfg := &qwen3_5.Config{
			HiddenSize: 4, NumAttentionHeads: 1, NumKeyValueHeads: 1,
			HeadDim: 4, RopeDim: 4, RopeTheta: 10000, Scale: 0.5,
			RMSNormEps: 1e-6, NoncausalFullAttention: true,
		}
		backbone := &qwen3_5.Model{
			Config: cfg, EmbedTokens: &nn.Embedding{Weight: mlx.FromValues([]float32{
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
			}, 3, 4)},
			Layers: layers, Norm: norm(),
		}
		readout := make([]float32, 255*4)
		readout[0], readout[5] = 1, 1
		m := &Model{Model: backbone, Readout: linear(readout, 255), config: config{Temperature: 2}}
		if caches := m.NewCaches(); caches != nil {
			t.Fatalf("noncausal model returned %d caches", len(caches))
		}
		b := &batch.Batch{
			InputIDs:   mlx.FromValues([]int32{0, 1, 2}, 1, 3),
			SeqOffsets: []int32{0}, SeqQueryLens: []int32{3},
		}
		noncausal, _ := m.Forward(b, nil)
		last := noncausal.Slice(mlx.Slice(), mlx.Slice(2, 3), mlx.Slice())
		got := m.scoreLogits(noncausal, 2, 2)
		lastValues := last.AsType(mlx.DTypeFloat32).Floats()
		if len(got) != 2 {
			t.Fatalf("readout returned %d logits, want 2", len(got))
		}
		if math.Abs(float64(got[0]-lastValues[0]/2)) > 1e-5 || math.Abs(float64(got[1]-lastValues[1]/2)) > 1e-5 {
			t.Fatalf("readout logits %v do not match the last hidden token %v", got, lastValues)
		}

		cfg.NoncausalFullAttention = false
		causal, _ := m.Forward(b, nil)
		causalLast := causal.Slice(mlx.Slice(), mlx.Slice(2, 3), mlx.Slice()).AsType(mlx.DTypeFloat32)
		mlx.Eval(causalLast)
		var drift float64
		for i, value := range causalLast.Floats() {
			drift += math.Abs(float64(value - lastValues[i]))
		}
		if drift < 1e-3 {
			t.Fatalf("two-layer noncausal forward matched causal forward (drift %g)", drift)
		}
	})
}
