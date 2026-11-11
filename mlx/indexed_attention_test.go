package mlx

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestIndexedBlockScaledDotProductAttentionMatchesGraph(t *testing.T) {
	for _, tc := range []struct {
		name                          string
		dtype                         DType
		headDim, keyLength, blockSize int
		causalTail                    bool
	}{
		{"complete blocks", DTypeBFloat16, 32, 64, 4, false},
		{"bf16 causal tails", DTypeBFloat16, 256, 37, 16, true},
		{"fp16 causal tails", DTypeFloat16, 256, 37, 16, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			withMLXThread(t, func(t *mlxthreadtest.T) {
				if !MetalIsAvailable() {
					t.Skip("requires Metal kernel")
				}
				Scoped(func() {
					const (
						batch       = 2
						queryHeads  = 4
						kvHeads     = 2
						queryLength = indexedBlockAttentionMinQueryLength
					)
					keyLength, headDim, blockSize := tc.keyLength, tc.headDim, tc.blockSize
					topK := keyLength / blockSize

					q := patternArray(tc.dtype, []int{batch, queryHeads, queryLength, headDim}, 0.1, 0.03, 7, 19)
					k := patternArray(tc.dtype, []int{batch, kvHeads, keyLength, headDim}, -0.05, 0.02, 5, 17)
					v := patternArray(tc.dtype, []int{batch, kvHeads, keyLength, headDim}, 0.2, 0.04, 3, 13)
					blockValues := make([]int32, batch*queryLength*topK)
					queryEndValues := make([]int32, batch*queryLength)
					for b := range batch {
						for row := range queryLength {
							end := keyLength
							if tc.causalTail {
								end = (row+b)%keyLength + 1
							}
							i := (b*queryLength + row) * topK
							for j := range topK {
								blockValues[i+j] = -1
								if j%2 == 0 && j/2 < end/blockSize {
									blockValues[i+j] = int32(j / 2)
								}
							}
							queryEndValues[b*queryLength+row] = int32(end)
						}
					}
					blocks := FromValues(blockValues, batch, queryLength, topK)
					queryEnds := FromValues(queryEndValues, batch, queryLength)
					scale := float32(1 / math.Sqrt(float64(headDim)))
					got := IndexedBlockScaledDotProductAttention(q, k, v, blocks, queryEnds, blockSize, scale)
					if got.name != "INDEXED_BLOCK_SCALED_DOT_PRODUCT_ATTENTION" {
						t.Fatalf("executed %s, want indexed Metal kernel", got.name)
					}
					got = got.AsType(DTypeFloat32)
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
		})
	}
}
