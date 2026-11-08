package dflash

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// With zero queries, attention must be the mean over the mathematically
// allowed values. This catches accidental causal masking of proposal rows
// and window-offset mistakes independently of transformer weight arithmetic.
func TestNoncausalWindowAndRollback(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		cfg, a := meanAttention(2, 4)
		kv := cache.NewRotatingKVCache(4)
		defer kv.Free()
		context := make([]float32, 16)
		for i := range 8 {
			context[2*i] = float32(i + 1)
			context[2*i+1] = float32(i + 1)
		}
		forward := func(offset int, contextValues []float32, proposals []float32, want []float32) {
			mlx.Scoped(func() {
				l := len(proposals) / 2
				n := len(contextValues) / 2
				bb := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, l), SeqOffsets: []int32{int32(offset)}, SeqQueryLens: []int32{int32(l)}}
				kv.PrepareSnapshots([]int{offset})
				out := a.Forward(mlx.FromValues(proposals, 1, l, 2), mlx.Zeros(mlx.DTypeFloat32, 1, 1, n, 2), mlx.FromValues(contextValues, 1, 1, n, 2), kv, bb, mlx.FromValues([]int32{int32(offset)}, 1), nn.AttentionMask{}, 1, int32(l), cfg)
				mlx.Eval(out)
				// MLX runs float32 matmuls as TF32 on M5-class GPUs by default,
				// up to ~7e-4 here against ~1e-7 with MLX_ENABLE_TF32=0. Mask or
				// rollback errors exceed 0.1% by orders of magnitude.
				for i, v := range out.Floats() {
					if math.Abs(float64(v-want[i])) > math.Max(1e-3, math.Abs(float64(want[i]))*1e-3) {
						t.Fatalf("offset %d index %d: got %g want %g", offset, i, v, want[i])
					}
				}
				snapshots := kv.TakeSnapshots()
				if !kv.Restore(nil, offset) && !kv.Restore(snapshots[0], offset) {
					t.Fatal("could not restore context")
				}
				for _, snapshot := range snapshots {
					if snapshot != nil {
						snapshot.Close()
					}
				}
				if kv.Offset() != offset {
					t.Fatal("proposal rows remained committed")
				}
			})
		}
		// At q=8: context 6,7,8 plus the whole proposal block 9,10 => 8.
		// At q=9: context 7,8 plus block 9,10 => 8.5.
		forward(8, context, []float32{9, 9, 10, 10}, []float32{8, 8, 8.5, 8.5})
		// Discarded proposal values must not contaminate the next context run.
		// The newly committed context row is 20, not the old proposal value 9.
		forward(9, []float32{20, 20}, []float32{30, 30, 40, 40}, []float32{21, 21, 24.5, 24.5})
	})
}

// Use a scalar visibility oracle across every acceptance position, with
// separately committed target context and rejected proposal values far apart.
func TestRepeatedBlockRollback(t *testing.T) {
	for _, width := range []int{2, 5, 8} {
		for _, initial := range []int{1, 7, 8, 9, 17} {
			for accepted := range width {
				mlxtest.RunSubtest(t, fmt.Sprintf("width %d context %d accepted %d", width, initial, accepted), func(t *mlxtest.T) {
					const window = 8
					cfg, a := meanAttention(2, window)
					kv := cache.NewRotatingKVCache(window)
					defer kv.Free()
					context := []float32{}
					commit := func(n int) {
						// One-token updates deliberately wrap the physical ring.
						for range n {
							i := len(context)
							v := float32((i*17)%31 + 1)
							context = append(context, v)
							b := &batch.Batch{SeqOffsets: []int32{int32(i)}}
							kv.Update(b, mlx.Zeros(mlx.DTypeFloat32, 1, 1, 1, 2), mlx.FromValues([]float32{v, v}, 1, 1, 1, 2))
						}
					}
					commit(initial)
					for round := range 3 {
						offset := len(context)
						kv.PrepareSnapshots([]int{offset})
						proposals := make([]float32, 2*width)
						for i := range width {
							proposals[2*i] = float32(100 + 13*i)
							proposals[2*i+1] = proposals[2*i]
						}
						b := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, width), SeqOffsets: []int32{int32(offset)}, SeqQueryLens: []int32{int32(width)}}
						out := a.Forward(mlx.FromValues(proposals, 1, width, 2), nil, nil, kv, b, mlx.FromValues([]int32{int32(offset)}, 1), nn.AttentionMask{}, 1, int32(width), cfg)
						mlx.Eval(out)
						if err := visibleMeanError(out.Floats(), context, proposals, offset, window); err != nil {
							t.Fatalf("width=%d initial=%d accepted=%d round=%d %v", width, initial, accepted, round, err)
						}
						snapshots := kv.TakeSnapshots()
						if len(snapshots) != 1 || snapshots[0] == nil {
							t.Fatal("missing rollback snapshot")
						}
						if !kv.Restore(snapshots[0], offset) {
							t.Fatal("rollback failed")
						}
						snapshots[0].Close()
						if kv.Offset() != offset {
							t.Fatal("wrong committed offset")
						}
						commit(accepted + 1)
					}
				})
			}
		}
	}
}

// The scalar oracle uses nonzero queries and checks both cache K/V and output,
// so skipping only context K, block K, or block Q cannot satisfy the contract.
func TestDraftPerLayerNoRoPE(t *testing.T) {
	for _, policy := range []struct {
		name, field string
		noRoPE      [2]bool
	}{
		{"omitted", "", [2]bool{}},
		{"empty", `,"no_rope_layers":[]`, [2]bool{}},
		{"full_nope", `,"no_rope_layers":[1]`, [2]bool{false, true}},
		{"sliding_nope", `,"no_rope_layers":[0]`, [2]bool{true, false}},
	} {
		for _, path := range []struct {
			name           string
			context, block bool
		}{
			{"context_only", true, false},
			{"block_only", false, true},
			{"combined", true, true},
		} {
			mlxtest.RunSubtest(t, policy.name+"/"+path.name, func(t *mlxtest.T) {
				mlx.Scoped(func() {
					m := newDraftRotaryTestModel(t, policy.field)
					caches := m.NewCaches()
					for _, c := range caches {
						defer c.Free()
					}
					var oracle draftRotaryOracle
					// Seed the real caches without invoking either rotary path.
					// Block-only must not fail because context maintenance is broken.
					seed := []float32{0.25, -0.5, 0.75, 0.25, -0.5, 1}
					for i, c := range caches {
						c.(cache.Attention).Update(&batch.Batch{SeqOffsets: []int32{0}},
							mlx.Zeros(mlx.DTypeFloat32, 1, 1, 3, 2), mlx.FromValues(seed, 1, 1, 3, 2))
						oracle.keys[i] = make([]float64, len(seed))
						for _, v := range seed {
							oracle.values[i] = append(oracle.values[i], float64(v))
						}
					}
					for round := range 2 {
						start := caches[0].Offset()
						b := &batch.Batch{SeqOffsets: []int32{int32(start)}}
						var context, block []float64
						if path.context {
							context = []float64{1, 2, -3, 1}
							b.Hidden = mlx.FromValues([]float32{1, 2, -3, 1}, 1, 2, 2)
						}
						if path.block {
							block = []float64{1, 2, -2, 1, 0.5, -1}
							b.InputIDs = mlx.FromValues([]int32{0, 1, 2}, 1, 3)
							b.SeqQueryLens = []int32{3}
						}
						want := oracle.forward(context, block, start, policy.noRoPE)
						hidden, aux := m.Forward(b, nil, caches)
						label := fmt.Sprintf("round %d offset %d", round, start)
						if !path.block {
							if hidden != nil || aux != nil {
								t.Fatal("context-only forward returned block hidden states")
							}
						} else {
							if hidden == nil || aux != hidden {
								t.Fatal("block forward did not return shared hidden/aux states")
							}
							checkDraftRotaryArray(t, label+" hidden", hidden, want)
						}
						for i, c := range caches {
							if wantOffset := len(oracle.keys[i]) / 2; c.Offset() != wantOffset {
								t.Fatalf("%s layer %d cache offset=%d want=%d", label, i, c.Offset(), wantOffset)
							}
							state := c.State()
							if len(state) != 2 {
								t.Fatalf("%s layer %d state roots=%d want=2", label, i, len(state))
							}
							checkDraftRotaryArray(t, fmt.Sprintf("%s layer %d keys", label, i), state[0], oracle.keys[i])
							checkDraftRotaryArray(t, fmt.Sprintf("%s layer %d values", label, i), state[1], oracle.values[i])
						}
					}
				})
			})
		}
	}
}

type windowedDraftTarget struct {
	*draftRotaryTarget
	window int
}

func (t *windowedDraftTarget) DraftWindow() int { return t.window }

func checkDraftWindowValues(t *mlxtest.T, label string, array *mlx.Array, want []float64) {
	t.Helper()
	array = mlx.Contiguous(array, false).AsType(mlx.DTypeFloat32)
	mlx.Eval(array)
	got := array.Floats()
	wantFloat := make([]float32, len(want))
	for i, v := range want {
		wantFloat[i] = float32(v)
	}
	slices.Sort(got)
	slices.Sort(wantFloat)
	if !slices.Equal(got, wantFloat) {
		t.Fatalf("%s values = %v, want %v", label, got, wantFloat)
	}
}

func TestDraftWindowCache(t *testing.T) {
	for _, tc := range []struct {
		name             string
		target           draftTarget
		initial, wrapped [][]float64
	}{
		{
			name: "default", target: &draftRotaryTarget{},
			initial: [][]float64{{2, 3}, {0, 1, 2, 3}},
			wrapped: [][]float64{{3, 4}, {0, 1, 2, 3, 4}},
		},
		{
			name: "bounded", target: &windowedDraftTarget{draftRotaryTarget: &draftRotaryTarget{}, window: 3},
			initial: [][]float64{{2, 3}, {1, 2, 3}},
			wrapped: [][]float64{{3, 4}, {2, 3, 4}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				m := &Model{
					Config: &Config{SlidingWindow: 2},
					Layers: []*Layer{{IsSliding: true}, {IsSliding: false}},
					target: tc.target,
				}
				caches := m.NewCaches()
				defer func() {
					for _, c := range caches {
						c.Free()
					}
				}()
				initial := &batch.Batch{SeqOffsets: []int32{0}, SeqQueryLens: []int32{4}}
				values := mlx.FromValues([]float32{0, 1, 2, 3}, 1, 1, 4, 1)
				for i, c := range caches {
					c.(cache.Attention).Update(initial, values, values)
					checkDraftWindowValues(t, fmt.Sprintf("initial cache %d", i), c.(cache.Attention).View(initial).V(), tc.initial[i])
					c.PrepareSnapshots([]int{4})
					next := &batch.Batch{SeqOffsets: []int32{4}, SeqQueryLens: []int32{1}}
					value := mlx.FromValue(float32(4))
					value = mlx.Reshape(value, 1, 1, 1, 1)
					c.(cache.Attention).Update(next, value, value)
					checkDraftWindowValues(t, fmt.Sprintf("wrapped cache %d", i), c.(cache.Attention).View(next).V(), tc.wrapped[i])
					snapshots := c.TakeSnapshots()
					if len(snapshots) != 1 {
						t.Fatalf("cache %d snapshots = %d", i, len(snapshots))
					}
					if !c.Restore(snapshots[0], 4) {
						t.Fatalf("cache %d could not restore", i)
					}
					if snapshots[0] != nil {
						snapshots[0].Close()
					}
					if c.Offset() != 4 {
						t.Fatalf("cache %d offset = %d", i, c.Offset())
					}
					checkDraftWindowValues(t, fmt.Sprintf("restored cache %d", i), c.(cache.Attention).View(initial).V(), tc.initial[i])
				}
			})
		})
	}
}

type draftRotaryTarget struct {
	model.Model
	embeddings *mlx.Array
}

func (m *draftRotaryTarget) TokenEmbeddings(ids *mlx.Array) *mlx.Array {
	return m.embeddings.TakeAxis(ids, 0)
}

func (*draftRotaryTarget) RawLogits(x *mlx.Array) *mlx.Array { return x }
func (*draftRotaryTarget) SetAuxHiddenLayers([]int)          {}
func (*draftRotaryTarget) NumLayers() int                    { return 2 }

func newDraftRotaryTestModel(t *mlxtest.T, field string) *Model {
	t.Helper()
	cfg, err := parseConfig(draftRotaryTestConfig(field))
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{
		Config: cfg, Layers: make([]*Layer, cfg.NumHiddenLayers), tensorPrefix: "draft.",
		target: &draftRotaryTarget{embeddings: mlx.FromValues([]float32{1, 2, -2, 1, 0.5, -1}, 3, 2)},
	}
	identity := mlx.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	ones := mlx.FromValues([]float32{1, 1}, 2)
	zero := mlx.Zeros(mlx.DTypeFloat32, 2, 2)
	tensors := map[string]*mlx.Array{
		"draft.fc.weight": identity, "draft.hidden_norm.weight": ones, "draft.norm.weight": ones,
	}
	for i := range m.Layers {
		prefix := fmt.Sprintf("draft.layers.%d.", i)
		for _, name := range []string{"q_proj", "k_proj", "v_proj", "o_proj"} {
			tensors[prefix+"self_attn."+name+".weight"] = identity
		}
		for _, name := range []string{"input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"} {
			tensors[prefix+name+".weight"] = ones
		}
		for _, name := range []string{"gate_proj", "up_proj", "down_proj"} {
			tensors[prefix+"mlp."+name+".weight"] = zero
		}
	}
	if err := m.LoadWeights(tensors); err != nil {
		t.Fatal(err)
	}
	return m
}

func checkDraftRotaryArray(t *mlxtest.T, label string, array *mlx.Array, want []float64) {
	t.Helper()
	// Cache state can be a strided view; Floats requires contiguous storage.
	array = mlx.Contiguous(array, false).AsType(mlx.DTypeFloat32)
	mlx.Eval(array)
	got := array.Floats()
	if len(got) != len(want) {
		t.Fatalf("%s length=%d want=%d", label, len(got), len(want))
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.Abs(float64(v)-want[i]) > 2e-3*math.Max(1, math.Abs(want[i])) {
			t.Errorf("%s[%d]: got %g want %g", label, i, v, want[i])
			return
		}
	}
}

// This two-wide scalar model has identity projections, unit norms, zero MLPs,
// and a window larger than its history. It does not call production RoPE/SDPA.
type draftRotaryOracle struct {
	keys, values [2][]float64
}

func draftRotaryNorm(x []float64) []float64 {
	out := make([]float64, len(x))
	for r := 0; r < len(x); r += 2 {
		inv := 1 / math.Sqrt((x[r]*x[r]+x[r+1]*x[r+1])/2+1e-6)
		out[r], out[r+1] = x[r]*inv, x[r+1]*inv
	}
	return out
}

func draftRotaryRotate(x []float64, start int) []float64 {
	out := slices.Clone(x)
	for r := 0; r < len(x); r += 2 {
		s, c := math.Sincos(float64(start + r/2))
		out[r], out[r+1] = x[r]*c-x[r+1]*s, x[r]*s+x[r+1]*c
	}
	return out
}

func (o *draftRotaryOracle) forward(context, block []float64, start int, noRoPE [2]bool) []float64 {
	contextValues := draftRotaryNorm(context)
	x := slices.Clone(block)
	for layer := range o.keys {
		contextKeys := draftRotaryNorm(contextValues)
		values := draftRotaryNorm(x)
		q := draftRotaryNorm(values)
		k := slices.Clone(q)
		if !noRoPE[layer] {
			contextKeys = draftRotaryRotate(contextKeys, start)
			q = draftRotaryRotate(q, start+len(context)/2)
			k = draftRotaryRotate(k, start+len(context)/2)
		}
		o.keys[layer] = append(o.keys[layer], contextKeys...)
		o.keys[layer] = append(o.keys[layer], k...)
		o.values[layer] = append(o.values[layer], contextValues...)
		o.values[layer] = append(o.values[layer], values...)
		for r := 0; r < len(x); r += 2 {
			scores := make([]float64, len(o.keys[layer])/2)
			for j := range scores {
				scores[j] = (q[r]*o.keys[layer][2*j] + q[r+1]*o.keys[layer][2*j+1]) / math.Sqrt(2)
			}
			maxScore := slices.Max(scores)
			total := 0.0
			for j, score := range scores {
				scores[j] = math.Exp(score - maxScore)
				total += scores[j]
			}
			for j, score := range scores {
				x[r] += score / total * o.values[layer][2*j]
				x[r+1] += score / total * o.values[layer][2*j+1]
			}
		}
	}
	return draftRotaryNorm(x)
}

// meanAttention is one head of zero queries over width-wide rows: every output
// row is the mean of the value rows it may see, and values copy their input.
func meanAttention(width int, window int32) (*Config, *Attention) {
	kv, ones := make([]float32, 2*width*width), make([]float32, width)
	for i := range width {
		kv[(width+i)*width+i], ones[i] = 1, 1
	}
	norm := func() *nn.RMSNorm { return nn.NewRMSNorm(mlx.FromValues(ones, width), 1e-6) }
	cfg := &Config{HiddenSize: int32(width), NumAttentionHeads: 1, NumKeyValueHeads: 1, HeadDim: int32(width), RMSNormEps: 1e-6, RopeTheta: 10000, Scale: float32(1 / math.Sqrt(float64(width))), SlidingWindow: window}
	return cfg, &Attention{
		QProj:  &nn.Linear{Weight: mlx.Zeros(mlx.DTypeFloat32, width, width)},
		KVProj: &nn.Linear{Weight: mlx.FromValues(kv, 2*width, width)},
		OProj:  &nn.Linear{Weight: mlx.FromValues(kv[width*width:], width, width)},
		QNorm:  norm(),
		KNorm:  norm(),
	}
}

// visibleMeanError checks two-wide meanAttention rows, each holding one value
// twice, against the mean of every proposal and the context values within
// window of the row's position start+q.
func visibleMeanError(out, context, proposals []float32, start, window int) error {
	rows := len(proposals) / 2
	for q := range rows {
		sum, count := 0.0, 0
		for k, v := range context {
			if start+q-k < window {
				sum += float64(v)
				count++
			}
		}
		for j := range rows {
			sum += float64(proposals[2*j])
			count++
		}
		want := sum / float64(count)
		for d := range 2 {
			if got := float64(out[2*q+d]); math.Abs(got-want) > math.Max(1e-3, math.Abs(want)*2e-3) {
				return fmt.Errorf("q=%d got=%g want=%g", q, got, want)
			}
		}
	}
	return nil
}
