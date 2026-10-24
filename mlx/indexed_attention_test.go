package mlx

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestIndexedBlockScaledDotProductAttentionMatchesGraph(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		Scoped(func() {
			const (
				batch       = 2
				queryHeads  = 4
				kvHeads     = 2
				queryLength = indexedBlockAttentionMinQueryLength
				keyLength   = 64
				headDim     = 32
				blockSize   = 4
				topK        = 16
			)

			q := patternArray(DTypeBFloat16, []int{batch, queryHeads, queryLength, headDim}, 0.1, 0.03, 7, 19)
			k := patternArray(DTypeBFloat16, []int{batch, kvHeads, keyLength, headDim}, -0.05, 0.02, 5, 17)
			v := patternArray(DTypeBFloat16, []int{batch, kvHeads, keyLength, headDim}, 0.2, 0.04, 3, 13)
			blockValues := make([]int32, batch*queryLength*topK)
			queryEndValues := make([]int32, batch*queryLength)
			for b := range batch {
				for row := range queryLength {
					i := (b*queryLength + row) * topK
					for j := range topK {
						blockValues[i+j] = -1
						if j%2 == 0 {
							blockValues[i+j] = int32(j / 2)
						}
					}
					queryEndValues[b*queryLength+row] = keyLength
				}
			}
			blocks := FromValues(blockValues, batch, queryLength, topK)
			queryEnds := FromValues(queryEndValues, batch, queryLength)
			const scale = float32(0.176776695)
			got := IndexedBlockScaledDotProductAttention(q, k, v, blocks, queryEnds, blockSize, scale).AsType(DTypeFloat32)
			want := indexedBlockScaledDotProductAttentionGraph(q, k, v, blocks, queryEnds, blockSize, scale).AsType(DTypeFloat32)
			Eval(got, want)

			gotValues, wantValues := got.Floats(), want.Floats()
			for i := range wantValues {
				if math.IsNaN(float64(gotValues[i])) || math.IsInf(float64(gotValues[i]), 0) {
					t.Fatalf("out[%d] = %v, want finite output", i, gotValues[i])
				}
				if diff := float32(math.Abs(float64(gotValues[i] - wantValues[i]))); diff > 0.005 {
					t.Fatalf("out[%d] = %v, want %v (diff %v)", i, gotValues[i], wantValues[i], diff)
				}
			}
		})
	})
}
