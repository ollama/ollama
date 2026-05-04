package turboquant

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestEncodeNonPowerOfTwoHeadDims is a defensive test for the dim-agnostic
// encode/decode path on the head dimensions that real ollama-supported
// models actually use:
//
//   - 192 — DeepSeek-V2 non-MLA: kqNopeHeadDim (128) + qkRopeHeadDim (64).
//   - 576 — DeepSeek-V3.1 / R1 MLA: kvLoraRank (512) + qkRopeHeadDim (64).
//   - 320 — sometimes-seen variant.
//
// The TurboQuant fused-FA CUDA kernel is template-instantiated only for
// head_dim ∈ {128} (Metal: {128, 256}); these models route through the
// DequantK + stock FA fallback path, which works at any head_dim.
//
// Both paths require the algorithm layer to encode/decode at arbitrary
// head_dim. This test pins that requirement: if anyone later inadvertently
// adds a power-of-2 / specific-dim assumption to the encoder or codebook
// caching, this test fails.
//
// The assertions are intentionally permissive — we only require non-trivial
// reconstruction (better than zeroing the vector). PPL or attention-quality
// gates for these models belong in the runtime-plumbing PR.
func TestEncodeNonPowerOfTwoHeadDims(t *testing.T) {
	for _, headDim := range []int{192, 320, 576} {
		for _, p := range []Preset{PresetTQ3, PresetTQ2, PresetTQ4} {
			t.Run(p.Name+"_d"+itoa(headDim), func(t *testing.T) {
				rng := rand.New(rand.NewPCG(uint64(headDim)*0x9e3779b9, 0xbf58476d))
				v := make([]float32, headDim)
				for i := range v {
					v[i] = float32(rng.NormFloat64())
				}

				ev, err := EncodeKeyVector(v, p)
				if err != nil {
					t.Fatalf("encode at dim=%d: %v", headDim, err)
				}
				raw, err := ev.MarshalBinary()
				if err != nil {
					t.Fatalf("marshal at dim=%d: %v", headDim, err)
				}
				rec, gotPreset, err := DecodeVector(raw)
				if err != nil {
					t.Fatalf("decode at dim=%d: %v", headDim, err)
				}
				if gotPreset.ID != p.ID {
					t.Fatalf("decode preset mismatch: got %d want %d", gotPreset.ID, p.ID)
				}
				if len(rec) != headDim {
					t.Fatalf("decode dim mismatch: got %d want %d", len(rec), headDim)
				}

				// Reconstruction must be non-trivial: better than zeroing.
				var origNorm, errNorm float64
				for i := range v {
					origNorm += float64(v[i]) * float64(v[i])
					d := float64(v[i] - rec[i])
					errNorm += d * d
				}
				origNorm = math.Sqrt(origNorm)
				errNorm = math.Sqrt(errNorm)
				if errNorm >= origNorm {
					t.Errorf("reconstruction worse than zero: ||v||=%.3f ||v-recon||=%.3f", origNorm, errNorm)
				}
			})
		}
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
