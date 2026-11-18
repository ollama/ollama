package qwen3_5

import (
	"fmt"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// Expose the ordinary cache contract so the same batched model forward keeps
// stock capture-all states. This is a numerical control, not a mock cache.
type capturedRecurrent struct{ cache.Recurrent }

func TestGatedDeltaCollectiveReplayModelRollback(t *testing.T) {
	for _, B := range []int{1, 4} {
		for _, L := range []int{1, 2, 3, 4, 5, 6, 7, 8} {
			for _, selection := range []string{"divergent", "full", "mixed"} {
				for _, precise := range []bool{false, true} {
					for _, dtType := range []mlx.DType{mlx.DTypeBFloat16, mlx.DTypeFloat32} {
						name := fmt.Sprintf("B%d/L%d/selection=%s/precise=%t/dt_bias=%v", B, L, selection, precise, dtType)
						mlxtest.RunSubtest(t, name, func(t *mlxtest.T) {
							const hidden, packed, value, heads, dim = 8, 256, 128, 4, 32
							cfg := &Config{
								HiddenSize: hidden, LinearNumKeyHeads: 2, LinearNumValueHeads: heads,
								LinearKeyHeadDim: dim, LinearValueHeadDim: dim, LinearConvKernelDim: 3,
								RMSNormEps: 1e-6, PreciseGatedDeltaDecay: precise,
							}
							g := &GatedDeltaNet{
								InProjQKVZ: nn.NewLinear(patternArray(packed+value, hidden), nil),
								InProjBA:   nn.NewLinear(patternArray(2*heads, hidden), nil),
								OutProj:    nn.NewLinear(patternArray(hidden, value), nil),
								Conv1D:     nn.NewConv1d(patternArray(packed, 3).ExpandDims(2), nil, 1, 0, 1, packed),
								NormWeight: mlx.AddScalar(mlx.Zeros(mlx.DTypeBFloat16, dim), 1),
								DtBias:     mlx.Zeros(dtType, heads),
								AExp:       mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, heads), .2),
							}
							sets := make([][]cache.Cache, 2)
							for side := range sets {
								for range B {
									owner := cache.NewRecurrentCache(2, packed, heads, dim, dim)
									defer owner.Free()
									sets[side] = append(sets[side], owner)
								}
							}
							input := func(length int) *mlx.Array {
								return patternArray(B*length, hidden).Reshape(B, length, hidden)
							}
							var pending *cache.SpeculativeBatch
							forward := func(side, length int, snapshots bool) *mlx.Array {
								b := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, B, length)}
								bundles := make([][]cache.Cache, B)
								for row, owner := range sets[side] {
									b.SeqOffsets = append(b.SeqOffsets, int32(owner.Offset()))
									b.SeqQueryLens = append(b.SeqQueryLens, int32(length))
									bundles[row] = []cache.Cache{owner}
									if snapshots {
										offsets := make([]int, length-1)
										for i := range offsets {
											offsets[i] = owner.Offset() + i + 1
										}
										owner.PrepareSnapshots(offsets)
									}
								}
								layers, err := cache.Batch(bundles)
								if err != nil {
									t.Fatal(err)
								}
								if side == 0 && snapshots {
									pending = cache.NewSpeculativeBatch(layers)
									if pending != nil {
										layers = pending.Layers
									}
								}
								var layer cache.Layer = layers[0]
								if side == 1 {
									layer = capturedRecurrent{layer.(cache.Recurrent)}
								}
								return g.Forward(input(length), b, layer, int32(B), int32(length), cfg)
							}
							// Seed a real model state, then verify from it with all graphs
							// and temporary inputs scoped to their originating forward.
							mlx.Scoped(func() { mlx.Eval(forward(0, 1, false), forward(1, 1, false)) })
							mlx.Scoped(func() {
								probeBatch := &batch.Batch{SeqQueryLens: make([]int32, B)}
								var conv, delta []*mlx.Array
								for row, owner := range sets[0] {
									probeBatch.SeqQueryLens[row] = int32(L)
									conv = append(conv, owner.State()[0])
									delta = append(delta, owner.State()[1])
								}
								prior := mlx.Concatenate(delta, 0)
								x := input(L)
								qkv := g.InProjQKVZ.Forward(x).Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, packed))
								projected, _ := nn.CausalConv1D(probeBatch, qkv, g.Conv1D, 2,
									nn.WithRecurrentState(mlx.Concatenate(conv, 0), prior), nn.WithConvSiLU())
								ba := g.InProjBA.Forward(x)
								if projected.DType() != mlx.DTypeBFloat16 || ba.DType() != mlx.DTypeBFloat16 ||
									g.DtBias.DType() != dtType || g.AExp.DType() != mlx.DTypeFloat32 || prior.DType() != mlx.DTypeFloat32 {
									t.Fatal("fixture does not exercise its declared GatedDelta dispatch")
								}
								if B == 4 && L == 8 && selection == "mixed" && !precise {
									t.Logf("dispatch native=%t packed=%v/%v BA=%v/%v DtBias=%v AExp=%v prior=%v/%v",
										dtType == mlx.DTypeBFloat16, projected.Dims(), projected.DType(), ba.Dims(), ba.DType(), g.DtBias.DType(), g.AExp.DType(), prior.Dims(), prior.DType())
								}
							})
							keeps, selected := make([]int, B), make([]int, B)
							prior := make([][]cache.Snapshot, 2)
							for side := range prior {
								prior[side] = make([]cache.Snapshot, B)
							}
							for row := range B {
								keep := 1 + 2*row%L
								switch selection {
								case "full":
									keep = L
								case "mixed":
									keep = []int{0, max(1, L/2), L, 1}[row%4]
								}
								keeps[row], selected[row] = keep, 1+keep
								if keep == 0 {
									for side := range sets {
										if side == 1 || B == 1 || L == 1 {
											prior[side][row] = sets[side][row].Snapshot(0)
										}
									}
								}
							}
							outputs := mlx.ScopedArrays(func() []*mlx.Array {
								return []*mlx.Array{forward(0, L, true), forward(1, L, true)}
							})
							assertBitEqual(t, "forward y", outputs[0], outputs[1])
							pending.Resolve(selected)
							pending.Close()
							for side, owners := range sets {
								for row, owner := range owners {
									keep := keeps[row]
									snaps := owner.TakeSnapshots()
									for i, snap := range snaps {
										if snap != nil && (keep == L || i != keep-1) {
											snap.Close()
											snaps[i] = nil
										}
									}
									var selected cache.Snapshot
									if keep == 0 {
										selected = prior[side][row]
									} else if keep < L {
										selected = snaps[keep-1]
									}
									if !owner.Restore(selected, 1+keep) {
										t.Fatalf("row %d failed restore to %d", row, 1+keep)
									}
									if selected != nil {
										selected.Close()
									}
								}
							}
							for row := range sets[0] {
								for i, state := range sets[0][row].State() {
									assertBitEqual(t, fmt.Sprintf("row%d/state%d", row, i), state, sets[1][row].State()[i])
								}
							}
							continued := mlx.ScopedArrays(func() []*mlx.Array {
								return []*mlx.Array{forward(0, 1, false), forward(1, 1, false)}
							})
							assertBitEqual(t, "continuation y", continued[0], continued[1])
						})
					}
				}
			}
		}
	}
}
