package mlxrunner

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

type visionGolden struct {
	Tag         string    `json:"tag"`
	Shape       []int     `json:"shape"`
	Mean        float64   `json:"mean"`
	Std         float64   `json:"std"`
	NormsHead   []float64 `json:"norms_head"`
	NormMean    float64   `json:"norm_mean"`
	Row0Head    []float64 `json:"row0_head"`
	RowLastHead []float64 `json:"rowlast_head"`
}

// gradientPNG renders the goldens' fixture: R=x%256, G=y%256, B=(x+y)%256.
// 480×336 at budget 70 has a budget-fill factor of exactly 1.0, so the
// resize is the identity and both sides consume identical patches.
func gradientPNG(t *testing.T, w, h int) []byte {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetNRGBA(x, y, color.NRGBA{R: uint8(x % 256), G: uint8(y % 256), B: uint8((x + y) % 256), A: 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// TestVisionGoldenParity compares EncodeVision on the real weights against
// the vendored mlx-vlm reference forward (testdata/gen_vision_goldens.py).
// Gated like the e2e smoke: OLLAMA_VISION_E2E=1 + local model. Regenerate:
//
//	uv run testdata/gen_vision_goldens.py gemma4:12b-nvfp4 testdata/vision_goldens_12b.json
func TestVisionGoldenParity(t *testing.T) {
	if os.Getenv("OLLAMA_VISION_E2E") == "" {
		t.Skip("set OLLAMA_VISION_E2E=1 to run vision golden parity")
	}
	modelName := os.Getenv("OLLAMA_VISION_E2E_MODEL")
	if modelName == "" {
		modelName = "gemma4:12b-nvfp4"
	}
	size := strings.TrimSuffix(strings.TrimPrefix(modelName, "gemma4:"), "-nvfp4")
	goldenPath := fmt.Sprintf("testdata/vision_goldens_%s.json", size)
	goldenData, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Skipf("golden %s not present: %v", goldenPath, err)
	}
	var golden visionGolden
	if err := json.Unmarshal(goldenData, &golden); err != nil {
		t.Fatal(err)
	}
	if root, err := model.Open(modelName); err != nil {
		t.Skipf("model %s not available: %v", modelName, err)
	} else {
		root.Close()
	}

	worker, err := mlxthread.Start("mlxrunner-golden", func() error {
		if err := mlx.CheckInit(); err != nil {
			return err
		}
		if mlx.GPUIsAvailable() {
			mlx.SetDefaultDeviceGPU()
		}
		return nil
	})
	if err != nil {
		t.Skipf("MLX not available: %v", err)
	}
	defer worker.Stop(context.Background(), func() {
		mlx.Sweep()
		mlx.ClearCache()
	})

	r := &Runner{Requests: make(chan Request), mlxThread: worker}
	if err := worker.Do(context.Background(), func() error {
		return r.Load(modelName)
	}); err != nil {
		t.Fatalf("load %s: %v", modelName, err)
	}
	vm := r.Model.(base.VisionModel)

	opts := api.DefaultOptions()
	opts.ImageMaxTokens = 70
	in, err := vm.NewVisionInput(gradientPNG(t, 480, 336), opts)
	if err != nil {
		t.Fatal(err)
	}
	if in.SoftTokens() != golden.Shape[0] {
		t.Fatalf("soft tokens = %d, golden %d", in.SoftTokens(), golden.Shape[0])
	}

	var feats []float32
	var dims []int
	if err := worker.Do(context.Background(), func() error {
		out := vm.EncodeVision(in).AsType(mlx.DTypeFloat32)
		mlx.Eval(out)
		dims = out.Dims()
		feats = out.Floats()
		return nil
	}); err != nil {
		t.Fatalf("EncodeVision: %v", err)
	}
	if len(dims) != 3 || dims[1] != golden.Shape[0] || dims[2] != golden.Shape[1] {
		t.Fatalf("shape = %v, golden %v", dims, golden.Shape)
	}

	n, d := golden.Shape[0], golden.Shape[1]
	var sum, sumSq float64
	norms := make([]float64, n)
	for i := 0; i < n; i++ {
		var rowSq float64
		for j := 0; j < d; j++ {
			v := float64(feats[i*d+j])
			sum += v
			sumSq += v * v
			rowSq += v * v
		}
		norms[i] = math.Sqrt(rowSq)
	}
	total := float64(n * d)
	mean := sum / total
	std := math.Sqrt(sumSq/total - mean*mean)
	var normMean float64
	for _, v := range norms {
		normMean += v
	}
	normMean /= float64(n)

	relDiff := func(a, b float64) float64 {
		den := math.Max(math.Abs(b), 1e-6)
		return math.Abs(a-b) / den
	}
	t.Logf("mean %.5f (golden %.5f)  std %.4f (golden %.4f)  norm_mean %.3f (golden %.3f)",
		mean, golden.Mean, std, golden.Std, normMean, golden.NormMean)

	if relDiff(normMean, golden.NormMean) > 0.02 {
		t.Errorf("norm_mean off by %.2f%%", 100*relDiff(normMean, golden.NormMean))
	}
	if relDiff(std, golden.Std) > 0.02 {
		t.Errorf("std off by %.2f%%", 100*relDiff(std, golden.Std))
	}
	if math.Abs(mean-golden.Mean) > 0.02*golden.Std {
		t.Errorf("mean %.5f vs golden %.5f exceeds 2%% of std", mean, golden.Mean)
	}
	for i, g := range golden.NormsHead {
		if relDiff(norms[i], g) > 0.03 {
			t.Errorf("norm[%d] = %.3f vs golden %.3f", i, norms[i], g)
		}
	}
	// Per-element bound: Go runs fused 4-bit quantized matmuls where the
	// reference matmuls dequantized bf16 weights, and 27 tower layers
	// accumulate the difference (observed ≤0.14 at element σ≈1.3-2.5, while
	// aggregates agree to 0.1%). Structural bugs — wrong norm, swapped axis,
	// missing scale — move elements by whole σ and fail this and the
	// aggregate checks together.
	var maxRowDelta float64
	checkRow := func(row int, want []float64) {
		for j, g := range want {
			v := float64(feats[row*d+j])
			delta := math.Abs(v - g)
			maxRowDelta = math.Max(maxRowDelta, delta)
			if delta > 0.15+0.05*math.Abs(g) {
				t.Errorf("row %d elem %d: %.4f vs golden %.4f", row, j, v, g)
			}
		}
	}
	checkRow(0, golden.Row0Head)
	checkRow(n-1, golden.RowLastHead)
	t.Logf("max sampled element delta: %.4f", maxRowDelta)
}
