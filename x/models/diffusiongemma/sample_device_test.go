package diffusiongemma

import (
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// TestSampleCanvasDeviceMatchesHost checks the on-device sampler reproduces the
// host reference for the deterministic outputs (argmax, entropy, self-cond
// top-k). The multinomial draw uses MLX's RNG, so it is only range-checked.
func TestSampleCanvasDeviceMatchesHost(t *testing.T) {
	skipIfNoMLX(t)

	const (
		L     = 5
		vocab = 64
		scK   = 4
		temp  = float32(0.8)
	)

	// Distinct, varied logits so argmax / top-k ordering is unambiguous.
	data := make([]float32, L*vocab)
	for i := range data {
		data[i] = float32(math.Sin(float64(i)*0.37))*3 + float32(math.Cos(float64(i)*0.131))*2
	}

	host := sampleCanvas(data, L, vocab, temp, scK, rand.New(rand.NewSource(1)))
	dev := sampleCanvasDevice(mlx.FromValues(data, L, vocab), L, vocab, temp, scK, mlx.RandomKey(42))

	for j := range L {
		if dev.argmax[j] != host.argmax[j] {
			t.Errorf("argmax[%d]: device=%d host=%d", j, dev.argmax[j], host.argmax[j])
		}
		if d := math.Abs(float64(dev.entropy[j] - host.entropy[j])); d > 0.02 {
			t.Errorf("entropy[%d]: device=%g host=%g (|d|=%g)", j, dev.entropy[j], host.entropy[j], d)
		}
		if dev.sampled[j] < 0 || int(dev.sampled[j]) >= vocab {
			t.Errorf("sampled[%d]=%d out of range", j, dev.sampled[j])
		}

		// Top-k probs as sorted-descending sets (device top-k is unordered).
		dp := append([]float32(nil), dev.scProbs[j*scK:(j+1)*scK]...)
		hp := append([]float32(nil), host.scProbs[j*scK:(j+1)*scK]...)
		sort.Slice(dp, func(a, b int) bool { return dp[a] > dp[b] })
		sort.Slice(hp, func(a, b int) bool { return hp[a] > hp[b] })
		for k := range scK {
			if d := math.Abs(float64(dp[k] - hp[k])); d > 1e-3 {
				t.Errorf("scProbs[%d] rank %d: device=%g host=%g", j, k, dp[k], hp[k])
			}
		}
	}
}
