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
	"github.com/ollama/ollama/mlxrunner/nn"
)

// Use aligned nontrivial projections, eight trained rows, two KV heads, and
// group-16 dynamic convolutions. Every weight is generated for this fixture.
func quantizedBatchModel(plain bool) *Model {
	const width, head, vocab = 256, 64, 128
	dtype := mlx.DTypeBFloat16
	if plain {
		dtype = mlx.DTypeFloat32
	}
	state := uint32(179497)
	values := func(n int, scale float32) []float32 {
		out := make([]float32, n)
		for i := range out {
			state ^= state << 13
			state ^= state >> 17
			state ^= state << 5
			out[i] = float32(int(state%2001)-1000) * scale / 1000
		}
		return out
	}
	linear := func(outputs, inputs int) nn.LinearLayer {
		w := mlx.FromValues(values(outputs*inputs, 0.06), outputs, inputs).AsType(mlx.DTypeBFloat16)
		q := nn.NewQuantizedLinear(w, nil, 16, 4, "nvfp4")
		scales := make([]float32, outputs)
		for i := range scales {
			scales[i] = float32(mlx.Nvfp4MaxProduct) * (0.75 + float32(i%3)*0.125)
		}
		q.GlobalScale = mlx.FromValues(scales, outputs)
		if plain {
			return nn.NewLinear(mlx.Dequantize(q.Weight, q.Scales, nil, 16, 4, "nvfp4", q.GlobalScale).AsType(dtype), nil)
		}
		return q
	}
	norm := func(n int) *nn.RMSNorm {
		values := make([]float32, n)
		for i := range values {
			values[i] = 1
		}
		return nn.NewRMSNorm(mlx.FromValues(values, n).AsType(dtype), 1e-6)
	}
	conv := func() *convolution {
		base := values(4*width, 0.015)
		for half := range 2 {
			for i := range width {
				base[half*2*width+i] += 1
			}
		}
		return &convolution{Base: mlx.FromValues(base, 2, 2, width).AsType(dtype), Projection: linear(64, width), Group: 16, Kernel: 2}
	}
	m := &Model{
		Config: &Config{
			HiddenSize: width, NumHiddenLayers: 2, NumAttentionHeads: 4, NumKeyValueHeads: 2,
			HeadDim: head, RMSNormEps: 1e-6, RopeTheta: 10000, Scale: 1.0 / 8, SlidingWindow: 17,
			BlockSize: 8, VocabSize: vocab, HasOwnLMHead: true,
		},
		target: &draftRotaryTarget{embeddings: mlx.FromValues(values(vocab*width, 0.75), vocab, width).AsType(dtype)},
		FC:     linear(width, width*2), LMHead: linear(vocab, width), HiddenNorm: norm(width), Norm: norm(width),
	}
	for i := range 2 {
		m.Layers = append(m.Layers, &Layer{
			InputNorm: norm(width), PostAttnNorm: norm(width), IsSliding: true,
			Attention: &Attention{
				QProj: linear(width, width), KVProj: linear(width, width), OProj: linear(width, width),
				QNorm: norm(head), KNorm: norm(head), NoRoPE: i == 1,
			},
			MLP:           &MLP{GateUpProj: linear(width*4, width), DownProj: linear(width, width*2)},
			AttentionConv: conv(), MLPConv: conv(),
		})
	}
	return m
}

type quantizedBatchDelta struct {
	count                                        int
	squared, referenceSquared, maxAbs, maxScaled float64
}

func (d *quantizedBatchDelta) compare(t *mlxtest.T, label string, got, want *mlx.Array, tolerance float64) {
	t.Helper()
	if err := d.compareArrays(got, want, tolerance); err != nil {
		t.Fatalf("%s: %v", label, err)
	}
}

func (d *quantizedBatchDelta) compareArrays(got, want *mlx.Array, tolerance float64) error {
	if got == nil || want == nil {
		return fmt.Errorf("missing numerical comparison: got %v, want %v", got, want)
	}
	if !slices.Equal(got.Dims(), want.Dims()) {
		return fmt.Errorf("shape %v, want %v", got.Dims(), want.Dims())
	}
	g := mlx.Contiguous(got, false).AsType(mlx.DTypeFloat32).Floats()
	w := mlx.Contiguous(want, false).AsType(mlx.DTypeFloat32).Floats()
	if len(g) == 0 || len(g) != len(w) {
		return fmt.Errorf("length %d, want nonempty length %d", len(g), len(w))
	}
	for i, v := range g {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.IsNaN(float64(w[i])) || math.IsInf(float64(w[i]), 0) {
			return fmt.Errorf("element %d is not finite: got %g, want %g", i, v, w[i])
		}
		delta := math.Abs(float64(v) - float64(w[i]))
		scaled := delta / math.Max(1, math.Abs(float64(w[i])))
		d.count++
		d.squared += delta * delta
		d.referenceSquared += float64(w[i]) * float64(w[i])
		d.maxAbs = math.Max(d.maxAbs, delta)
		d.maxScaled = math.Max(d.maxScaled, scaled)
		if scaled > tolerance {
			return fmt.Errorf("element %d got=%g want=%g scaled delta=%g exceeds %g", i, v, w[i], scaled, tolerance)
		}
	}
	return nil
}

func TestQuantizedBatchDeltaRejectsInvalidComparisons(t *testing.T) {
	for _, tc := range []struct {
		name                string
		got, want           []float32
		gotShape, wantShape []int
		valid               bool
	}{
		{"finite equal", []float32{1, 2}, []float32{1, 2}, []int{1, 2}, []int{1, 2}, true},
		{"finite unequal", []float32{1, 2}, []float32{1, 3}, []int{2}, []int{2}, false},
		{"empty", nil, nil, []int{0}, []int{0}, false},
		{"shape", []float32{1, 2}, []float32{1, 2}, []int{1, 2}, []int{2, 1}, false},
		{"length", []float32{1}, []float32{1, 2}, []int{1}, []int{2}, false},
		{"matching nan", []float32{float32(math.NaN())}, []float32{float32(math.NaN())}, []int{1}, []int{1}, false},
		{"matching infinity", []float32{float32(math.Inf(1))}, []float32{float32(math.Inf(1))}, []int{1}, []int{1}, false},
		{"nan reference", []float32{1}, []float32{float32(math.NaN())}, []int{1}, []int{1}, false},
	} {
		mlxtest.RunSubtest(t, tc.name, func(t *mlxtest.T) {
			array := func(values []float32, shape []int) *mlx.Array {
				if len(values) == 0 {
					return mlx.Zeros(mlx.DTypeFloat32, shape...)
				}
				return mlx.FromValues(values, shape...)
			}
			got, want := array(tc.got, tc.gotShape), array(tc.want, tc.wantShape)
			d := &quantizedBatchDelta{}
			err := d.compareArrays(got, want, .02)
			if (err == nil) != tc.valid {
				t.Fatalf("comparison error %v, want valid=%t", err, tc.valid)
			}
		})
	}
}

func TestDraftQuantizedBatchContexts(t *testing.T) {
	// The FP32 reference requires full-precision matmuls across batch shapes.
	t.Setenv("MLX_ENABLE_TF32", "0")
	for _, plain := range []bool{true, false} {
		for _, rows := range []int{1, 4} {
			mlxtest.RunSubtest(t, fmt.Sprintf("dequantized_f32=%v/B%d", plain, rows), func(t *mlxtest.T) {
				if !mlx.GPUIsAvailable() {
					t.Skip("NVFP4 fixture requires GPU")
				}
				m := quantizedBatchModel(plain)
				dtype, tolerance := mlx.DTypeBFloat16, 0.02
				if plain {
					dtype, tolerance = mlx.DTypeFloat32, 0.0002
				}
				features := func(row, count, round int) *mlx.Array {
					if count == 0 {
						return nil
					}
					v := make([]float32, count*512)
					for i := range v {
						v[i] = float32(((i+31*row+11*round)*37)%251-125) / 128
					}
					return mlx.FromValues(v, 1, count, 512).AsType(dtype)
				}
				serial, together := make([][]cache.Cache, rows), make([][]cache.Cache, rows)
				for i := range rows {
					serial[i], together[i] = m.NewCaches(), m.NewCaches()
					for _, caches := range [][]cache.Cache{serial[i], together[i]} {
						for _, c := range caches {
							defer c.Free()
						}
						seed := []int{0, 3, 9, 25}[i]
						if seed > 0 {
							m.Forward(&batch.Batch{SeqOffsets: []int32{0}, Hidden: features(i, seed, 0)}, nil, cache.Layers(caches))
						}
					}
				}
				deltas := map[string]*quantizedBatchDelta{"hidden": {}, "logits": {}, "state": {}, "restored": {}}
				for round, counts := range [][]int{{2, 0, 1, 8}, {0, 3, 8, 2}, {1, 2, 0, 4}, {0, 0, 0, 0}} {
					inputs, want := make([]*batch.Batch, rows), make([]*mlx.Array, rows)
					for i := range rows {
						offset := serial[i][0].Offset()
						ids := []int32{int32(7 + i*11), 5, 9, 17, 25, 41, 63, 89}
						inputs[i] = &batch.Batch{InputIDs: mlx.FromValues(ids, 1, 8), SeqOffsets: []int32{int32(offset)}, SeqQueryLens: []int32{8}, Hidden: features(i, counts[i], round+1)}
						for _, caches := range [][]cache.Cache{serial[i], together[i]} {
							for _, c := range caches {
								c.PrepareSnapshots([]int{offset + counts[i]})
							}
						}
						want[i], _ = m.Forward(inputs[i], nil, cache.Layers(serial[i]))
					}
					got := m.ForwardBatch(inputs, together)
					logits := m.Unembed(got)
					for i := range rows {
						label := fmt.Sprintf("round%d row%d", round, i)
						deltas["hidden"].compare(t, label+" hidden", got.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice()), want[i], tolerance)
						deltas["logits"].compare(t, label+" logits", logits.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice()), m.Unembed(want[i]), tolerance)
						for layer := range m.Layers {
							a, b := together[i][layer], serial[i][layer]
							if a.Offset() != b.Offset() {
								t.Fatalf("%s offset %d want %d", label, a.Offset(), b.Offset())
							}
							for j, v := range a.State() {
								deltas["state"].compare(t, label+" cache", v, b.State()[j], tolerance)
							}
							boundary := int(inputs[i].SeqOffsets[0]) + counts[i]
							for _, c := range []cache.Cache{a, b} {
								s := c.TakeSnapshots()
								if len(s) != 1 {
									t.Fatalf("%s snapshots %d want 1", label, len(s))
								}
								if !c.Restore(nil, boundary) && !c.Restore(s[0], boundary) {
									t.Fatalf("%s failed restore", label)
								}
								if s[0] != nil {
									s[0].Close()
								}
							}
							for j, v := range a.State() {
								deltas["restored"].compare(t, label+" restored", v, b.State()[j], tolerance)
							}
						}
					}
				}
				for _, name := range []string{"hidden", "logits", "state", "restored"} {
					d := deltas[name]
					t.Logf("%s count=%d max_abs=%g max_scaled=%g relative_l2=%g", name, d.count, d.maxAbs, d.maxScaled, math.Sqrt(d.squared/math.Max(d.referenceSquared, 1e-30)))
				}
			})
		}
	}
}

func TestDraftProjectionShapeDiagnostic(t *testing.T) {
	for _, plain := range []bool{true, false} {
		mlxtest.RunSubtest(t, fmt.Sprintf("dequantized_f32=%v", plain), func(t *mlxtest.T) {
			if !mlx.GPUIsAvailable() {
				t.Skip("NVFP4 fixture requires GPU")
			}
			m := quantizedBatchModel(plain)
			dtype := mlx.DTypeBFloat16
			if plain {
				dtype = mlx.DTypeFloat32
			}
			for _, count := range []int{1, 2, 3, 8} {
				values := make([]float32, count*512)
				for i := range values {
					values[i] = float32(((i+62+11)*37)%251-125) / 128
				}
				x := mlx.FromValues(values, 1, count, 512).AsType(dtype)
				padded := mlx.PadConstant(x, []int{1}, []int{0}, []int{8 - count})
				four := mlx.Concatenate([]*mlx.Array{padded, padded, padded, padded}, 0)
				serial := m.FC.Forward(x)
				together := m.FC.Forward(four).Slice(mlx.Slice(2, 3), mlx.Slice(0, count), mlx.Slice())
				for _, pair := range []struct {
					name string
					a, b *mlx.Array
				}{
					{"fc", together, serial},
					{"fc_norm", m.HiddenNorm.Forward(together, 1e-6), m.HiddenNorm.Forward(serial, 1e-6)},
				} {
					d := &quantizedBatchDelta{}
					d.compare(t, pair.name, pair.a, pair.b, 0.02)
					t.Logf("context=%d %s count=%d max_abs=%g max_scaled=%g relative_l2=%g", count, pair.name, d.count, d.maxAbs, d.maxScaled, math.Sqrt(d.squared/math.Max(d.referenceSquared, 1e-30)))
				}
			}
		})
	}
}

func TestDraftContextStagesDiagnostic(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.GPUIsAvailable() {
			t.Skip("NVFP4 fixture requires GPU")
		}
		m := quantizedBatchModel(true)
		rows := make([]*mlx.Array, 4)
		var single *mlx.Array
		for row, count := range []int{2, 0, 1, 8} {
			v := make([]float32, count*512)
			for i := range v {
				v[i] = float32(((i+31*row+11)*37)%251-125) / 128
			}
			if count == 0 {
				rows[row] = mlx.Zeros(mlx.DTypeFloat32, 1, 8, 512)
				continue
			}
			x := mlx.FromValues(v, 1, count, 512)
			if row == 2 {
				single = x
			}
			rows[row] = mlx.PadConstant(x, []int{1}, []int{0}, []int{8 - count})
		}
		full := mlx.Concatenate(rows, 0)
		fcA, fcB := m.FC.Forward(full), m.FC.Forward(single)
		hA, hB := m.HiddenNorm.Forward(fcA, 1e-6), m.HiddenNorm.Forward(fcB, 1e-6)
		three := func(x *mlx.Array) *mlx.Array { return x.Slice(mlx.Slice(2, 3), mlx.Slice(0, 1), mlx.Slice()) }
		four := func(x *mlx.Array) *mlx.Array {
			return x.Slice(mlx.Slice(2, 3), mlx.Slice(), mlx.Slice(0, 1), mlx.Slice())
		}
		compare := func(label string, a, b *mlx.Array) {
			d := &quantizedBatchDelta{}
			d.compare(t, label, a, b, 0.02)
			t.Logf("%s dtype=%v count=%d max_abs=%g max_scaled=%g relative_l2=%g", label, a.DType(), d.count, d.maxAbs, d.maxScaled, math.Sqrt(d.squared/math.Max(d.referenceSquared, 1e-30)))
		}
		// Build the actual context path before materializing any intermediates.
		positionsA, positionsB := mlx.FromValues([]int32{0, 3, 9, 25}, 4), mlx.FromValues([]int32{9}, 1)
		type pair struct {
			label string
			a, b  *mlx.Array
		}
		pairs := []pair{{"fc", three(fcA), fcB}, {"hidden_norm", three(hA), hB}}
		for i, l := range m.Layers {
			kvA, kvB := l.Attention.KVProj.Forward(hA), l.Attention.KVProj.Forward(hB)
			kA, vA := l.Attention.contextKV(hA, positionsA, m.Config)
			kB, vB := l.Attention.contextKV(hB, positionsB, m.Config)
			pairs = append(pairs, pair{fmt.Sprintf("layer%d kv_projection", i), three(kvA), kvB}, pair{fmt.Sprintf("layer%d context_k", i), four(kA), kB}, pair{fmt.Sprintf("layer%d context_v", i), four(vA), vB})
		}
		for _, p := range pairs {
			compare(p.label, p.a, p.b)
		}
	})
}
