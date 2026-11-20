package mlx

import (
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestRelativePositionAttentionBF16Probabilities(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if !hasMetalTensorOps() {
			t.Skip("requires Metal tensor ops")
		}
		const queries, keys, width, window = 17, 273, 128, 176
		const scale = 0.25
		// Each output channel samples one key's individual contribution. The
		// peaks straddle absolute block boundaries and change along the window.
		positions := []int{84, 93, 124, 127, 130, 190, 199, 240, 250, 255, 256, 265, 271}
		keyScore := make([]float32, keys)
		keyScore[124], keyScore[130], keyScore[190] = 5.5, 1.1875, 4
		keyScore[240], keyScore[265] = 2.5, 1.25
		qData := make([]float32, queries*width)
		kData := make([]float32, keys*width)
		vData := make([]float32, keys*width)
		sinkData := make([]float32, queries)
		for i := range queries {
			qData[i*width] = 1
			sinkData[i] = float32(i%3) + 3.5
		}
		for key, score := range keyScore {
			kData[key*width] = score / scale
		}
		for channel, key := range positions {
			vData[key*width+channel] = 64
		}
		relative := []float32{0.25, 0, 0, -0.5, 0, 0, 0, 0, 0.75}
		q := FromValues(qData, 1, 1, queries, width)
		k := FromValues(kData, 1, 1, keys, width).AsType(DTypeBFloat16)
		v := FromValues(vData, 1, 1, keys, width).AsType(DTypeBFloat16)
		table := BroadcastTo(FromValues(relative, 1, 1, 1, len(relative)).AsType(DTypeBFloat16), 1, 1, queries, int32(len(relative)))
		sink := FromValues(sinkData, 1, 1, queries, 1).AsType(DTypeBFloat16)
		for _, firstKey := range []int{0, 61} {
			out := RelativePositionAttentionBF16Probabilities(q, k, v, table, sink, firstKey, window, scale)
			if out == nil || out.DType() != DTypeFloat32 {
				t.Fatal("expected the FP32 attention result")
			}
			actual := Contiguous(out, false).Floats()
			for row := range queries {
				end := keys - queries + row + 1
				start := max(0, end-window)
				scores := make([]float64, end)
				denominator := math.Exp(float64(sinkData[row]))
				for key := start; key < end; key++ {
					score := float64(keyScore[key])
					if dist := end - 1 - key; dist < len(relative) {
						score += float64(relative[dist])
					}
					scores[key] = score
					denominator += math.Exp(score)
				}
				for channel, key := range positions {
					want := 0.0
					if start <= key && key < end {
						// At this key's multiply, the running maximum covers
						// its absolute 128-key block and all more recent blocks.
						blockStart := (firstKey+key)/128*128 - firstKey
						peak := slices.Max(scores[max(start, blockStart):])
						p := float32(math.Exp(scores[key] - peak))
						// Round just the unnormalized numerator weight to nearest
						// even BF16. The denominator above retains full precision.
						bits := math.Float32bits(p)
						p = math.Float32frombits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
						want = 64 * float64(p) * math.Exp(peak) / denominator
					}
					got := float64(actual[row*width+channel])
					if math.IsNaN(got) || math.Abs(got-want) > 2e-6*max(1.0, math.Abs(want)) {
						t.Fatalf("first key %d, query %d, value key %d: got %.9g, want %.9g", firstKey, row, key, got, want)
					}
				}
			}
		}
	})
}

func TestRelativePositionAttentionBF16ProbabilitySink(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if !hasMetalTensorOps() {
			t.Skip("requires Metal tensor ops")
		}
		const queries, keys, width = 16, 1024, 128
		q := Zeros(DTypeFloat32, 1, 1, queries, width)
		k := Zeros(DTypeBFloat16, 1, 1, keys, width)
		v := BroadcastTo(FromValue(float32(1)).AsType(DTypeBFloat16), 1, 1, keys, width)
		table := Zeros(DTypeBFloat16, 1, 1, queries, 1)
		sink := BroadcastTo(FromValue(float32(92)).AsType(DTypeBFloat16), 1, 1, queries, 1)
		out := RelativePositionAttentionBF16Probabilities(q, k, v, table, sink, 0, keys, 0.125)
		if out == nil {
			t.Fatal("expected the BF16 probability kernel")
		}
		// Every visible value is one. exp(92) overflows FP32, even though
		// the sink-gated mean for 1024 visible keys is a normal FP32 value.
		want := 1 / (1 + math.Exp(92-math.Log(keys)))
		raw := Contiguous(out, false).Floats()
		if got := float64(raw[len(raw)-1]); math.IsNaN(got) || math.Abs(got/want-1) > 2e-5 {
			t.Fatalf("last query with large sink: got %.9g, want %.9g", got, want)
		}
	})
}
