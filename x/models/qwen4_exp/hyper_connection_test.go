package qwen4_exp

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

func TestHyperConnectionMatchesReferenceFormula(t *testing.T) {
	mlxtest.Setup(t)

	cfg := &Config{HCCount: 2, HiddenSize: 2, HCLowRank: 2, RMSNormEps: 1e-6}
	normWeight := []float32{1.1, 0.9, 1.2, 0.8}
	downWeight := [][]float32{{0.2, -0.3, 0.4, 0.1}, {-0.1, 0.5, 0.2, -0.4}}
	upWeight := [][]float32{{0.3, -0.2}, {0.1, 0.4}, {-0.5, 0.2}, {0.25, 0.15}}
	injectWeight := [][]float32{{0.2, -0.1, 0.3, 0.4}, {-0.3, 0.2, 0.1, 0.5}}
	input := []float32{1, 2, 3, 4}

	h := &hyperConnection{
		Norm:         &streamRMSNorm{Weight: mlx.FromValues(normWeight, 2, 2)},
		InputMixDown: nn.NewLinear(matrix(downWeight), nil),
		InputMixUp:   nn.NewLinear(matrix(upWeight), nil),
		BlockInject:  nn.NewLinear(matrix(injectWeight), nil),
	}
	residual := mlx.FromValues(input, 1, 1, 4)
	branch, state := h.Prepare(residual, cfg)
	got := h.Inject(state, branch, cfg).AsType(mlx.DTypeFloat32)
	reduced := h.Reduce(residual, cfg).AsType(mlx.DTypeFloat32)
	mlx.Eval(got, reduced)

	normed := append([]float32(nil), input...)
	for stream := range int(cfg.HCCount) {
		start := stream * int(cfg.HiddenSize)
		var square float64
		for i := range int(cfg.HiddenSize) {
			x := float64(input[start+i])
			square += x * x
		}
		invRMS := 1 / math.Sqrt(square/float64(cfg.HiddenSize)+float64(cfg.RMSNormEps))
		for i := range int(cfg.HiddenSize) {
			normed[start+i] = float32(float64(input[start+i]) * invRMS * float64(normWeight[start+i]))
		}
	}

	down := matvec(downWeight, normed)
	for i := range down {
		down[i] /= float32(cfg.HCCount)
		down[i] *= 1 / (1 + float32(math.Exp(float64(-down[i]))))
	}
	mix := matvec(upWeight, down)
	wantBranch := make([]float32, cfg.HiddenSize)
	for stream := range int(cfg.HCCount) {
		for i := range int(cfg.HiddenSize) {
			j := stream*int(cfg.HiddenSize) + i
			gate := 1 / (1 + float32(math.Exp(float64(-mix[j]))))
			wantBranch[i] += gate * normed[j] / float32(cfg.HCCount)
		}
	}

	injection := matvec(injectWeight, normed)
	want := append([]float32(nil), input...)
	for stream := range int(cfg.HCCount) {
		weight := 2 / (1 + float32(math.Exp(float64(-injection[stream]/float32(cfg.HCCount)))))
		for i := range int(cfg.HiddenSize) {
			want[stream*int(cfg.HiddenSize)+i] += weight * wantBranch[i]
		}
	}

	assertClose(t, "mixed branch", reduced.Floats(), wantBranch)
	assertClose(t, "injected streams", got.Floats(), want)
}

func matrix(rows [][]float32) *mlx.Array {
	values := make([]float32, 0, len(rows)*len(rows[0]))
	for _, row := range rows {
		values = append(values, row...)
	}
	return mlx.FromValues(values, len(rows), len(rows[0]))
}

func matvec(weight [][]float32, input []float32) []float32 {
	output := make([]float32, len(weight))
	for i, row := range weight {
		for j, value := range row {
			output[i] += value * input[j]
		}
	}
	return output
}

func assertClose(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("%s[%d] = %v, want %v", name, i, got[i], want[i])
		}
	}
}
