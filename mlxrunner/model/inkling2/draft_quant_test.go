package inkling2

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func draftTestValues(rng *rand.Rand, scale float64, shape ...int) *mlx.Array {
	size := 1
	for _, d := range shape {
		size *= d
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(rng.NormFloat64() * scale)
	}
	return mlx.FromValues(data, shape...).AsType(mlx.DTypeBFloat16)
}

func requireDrift(t *mlxtest.T, name string, got, want *mlx.Array, maxRelativeL2 float64) {
	t.Helper()
	if !slices.Equal(got.Dims(), want.Dims()) {
		t.Fatalf("%s: shape %v, want %v", name, got.Dims(), want.Dims())
	}
	var diffSq, wantSq float64
	g, w := got.AsType(mlx.DTypeFloat32).Flatten(0, -1).Floats(), want.AsType(mlx.DTypeFloat32).Flatten(0, -1).Floats()
	for i := range w {
		d := float64(g[i]) - float64(w[i])
		diffSq += d * d
		wantSq += float64(w[i]) * float64(w[i])
	}
	relative := math.Sqrt(diffSq / wantSq)
	t.Logf("%s: relative L2 drift %.2e", name, relative)
	if !(relative <= maxRelativeL2) {
		t.Errorf("%s: relative L2 drift %.2e exceeds %.0e", name, relative, maxRelativeL2)
	}
}

// requireIdentical fails unless got and want hold the same bits.
func requireIdentical(t *mlxtest.T, name string, got, want *mlx.Array) {
	t.Helper()
	if got.DType() != want.DType() || !slices.Equal(got.Dims(), want.Dims()) {
		t.Fatalf("%s: %v %v, want %v %v", name, got.DType(), got.Dims(), want.DType(), want.Dims())
	}
	g := mlx.Contiguous(got.AsType(mlx.DTypeFloat32), false).Floats()
	w := mlx.Contiguous(want.AsType(mlx.DTypeFloat32), false).Floats()
	for i := range w {
		if math.Float32bits(g[i]) != math.Float32bits(w[i]) {
			t.Fatalf("%s: value %d is %g, want %g", name, i, g[i], w[i])
		}
	}
}

// requireClose fails unless max|got-want| is within fraction of max|want|.
func requireClose(t *mlxtest.T, name string, got, want *mlx.Array, fraction float64) {
	t.Helper()
	if !slices.Equal(got.Dims(), want.Dims()) {
		t.Fatalf("%s: shape %v, want %v", name, got.Dims(), want.Dims())
	}
	got, want = got.AsType(mlx.DTypeFloat32), want.AsType(mlx.DTypeFloat32)
	diff := float64(mlx.Sub(got, want).Abs().Reshape(-1).MaxAxis(0, false).Float())
	scale := float64(want.Abs().Reshape(-1).MaxAxis(0, false).Float())
	if !(diff <= scale*fraction) {
		t.Fatalf("%s: largest difference %g for values up to %g", name, diff, scale)
	}
}

// Stacked projections split back into the stock product of each weight. BF16
// weights differ only in FP32 reduction order. Quantized weights match the
// FP32 product of their dequantized values within the BF16 roundings of the
// result and, in tiled kernels, of the dequantized weights. 8-bit weights also
// stay within their quantization error of the BF16 product.
func TestDraftProjectionMatchesStock(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			rng := rand.New(rand.NewPCG(7, 23))
			const width = 512
			weights := []*mlx.Array{draftTestValues(rng, 0.03, 384, width), draftTestValues(rng, 0.03, 128, width), draftTestValues(rng, 0.03, 128, width)}
			for _, rows := range []int{1, 3, 8} {
				x := draftTestValues(rng, 1, 1, rows, width)
				for _, bits := range []int{0, 8, draftVocabularyBits} {
					p := newDraftProjection(bits, weights...)
					if (p.Scales != nil) != (bits > 0) || p.bits != bits {
						t.Fatalf("%d bits: projection holds %d-bit weights, quantized %t", bits, p.bits, p.Scales != nil)
					}
					got := p.forward(x)
					var dequantized *mlx.Array
					if bits > 0 {
						dequantized = mlx.Dequantize(p.Weight, p.Scales.AsType(mlx.DTypeFloat32), p.Biases.AsType(mlx.DTypeFloat32), draftGroupSize, bits, "affine", nil)
					}
					start := 0
					for i, w := range weights {
						name := fmt.Sprintf("%d rows, %d bits, projection %d", rows, bits, i)
						stock := x.Matmul(w.Transpose(1, 0))
						if dequantized == nil {
							requireDrift(t, name, got[i], stock, 4e-3)
							continue
						}
						rowsOf := dequantized.Slice(mlx.Slice(start, start+w.Dim(0)), mlx.Slice())
						start += w.Dim(0)
						requireDrift(t, name, got[i], x.AsType(mlx.DTypeFloat32).Matmul(rowsOf.Transpose(1, 0)), 6e-3)
						if bits == 8 {
							requireDrift(t, name+" against BF16", got[i], stock, 2e-2)
						}
					}
				}
			}
		})
	})
}

// Affine groups cannot tile an input width that is not a multiple of
// draftGroupSize, so such a projection keeps its BF16 weights.
func TestDraftProjectionKeepsUntiledWidths(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			rng := rand.New(rand.NewPCG(13, 17))
			const width = draftGroupSize + 32
			weights := []*mlx.Array{draftTestValues(rng, 0.03, 64, width), draftTestValues(rng, 0.03, 32, width)}
			p := newDraftProjection(draftBits, weights...)
			if p.Scales != nil || p.Biases != nil || p.bits != 0 {
				t.Fatalf("width %d: projection holds %d-bit weights, quantized %t", width, p.bits, p.Scales != nil)
			}
			x := draftTestValues(rng, 1, 1, 3, width)
			got := p.forward(x)
			for i, w := range weights {
				requireDrift(t, fmt.Sprintf("projection %d", i), got[i], x.Matmul(w.Transpose(1, 0)), 4e-3)
			}
		})
	})
}

// One product with the tap weights forms every mixture of the per-mixture
// weighted sums up to FP32 summation order.
func TestDraftMixesMatchWeightedSums(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			rng := rand.New(rand.NewPCG(3, 41))
			const width, taps, layers, rows = 256, 4, 3, 3
			d := &draft{
				config:        draftConfig{Width: width, Taps: make([]int, taps), Epsilon: 1e-6},
				Fuse:          mlx.SoftmaxAxis(draftTestValues(rng, 1, layers+1, taps).AsType(mlx.DTypeFloat32), -1, true).AsType(mlx.DTypeBFloat16).AsType(mlx.DTypeFloat32),
				RecurrentNorm: mlx.AddScalar(draftTestValues(rng, 0.1, width).AsType(mlx.DTypeFloat32), 1),
				Layers:        make([]draftLayer, layers),
			}
			for i := range d.Layers {
				d.Layers[i].ContextNorm = mlx.AddScalar(draftTestValues(rng, 0.1, width).AsType(mlx.DTypeFloat32), 1)
			}
			features := draftTestValues(rng, 2, 1, rows, taps*width)
			got := d.mixes(features)
			x := features.Reshape(1, rows, taps, width).AsType(mlx.DTypeFloat32)
			var norms []*mlx.Array
			for _, l := range d.Layers {
				norms = append(norms, l.ContextNorm)
			}
			for i, norm := range append(norms, d.RecurrentNorm) {
				weights := d.Fuse.Slice(mlx.Slice(i, i+1), mlx.Slice()).Reshape(1, 1, taps, 1)
				want := modelRMS(mlx.Sum(mlx.Mul(x, weights), 2, false), norm, 1e-6)
				requireDrift(t, fmt.Sprintf("mixture %d", i), got[i], want, 4e-3)
			}
		})
	})
}

// The compiled residual sum, recurrent gate and score correction keep the
// rounding of the separate ops they replace.
func TestDraftCompiledStepsMatchEager(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.EnableCompile()
		mlx.Scoped(func() {
			rng := rand.New(rand.NewPCG(11, 5))
			same := func(name string, got, want *mlx.Array) {
				t.Helper()
				if !slices.Equal(got.AsType(mlx.DTypeFloat32).Flatten(0, -1).Floats(), want.AsType(mlx.DTypeFloat32).Flatten(0, -1).Floats()) {
					t.Errorf("%s differs from the eager ops", name)
				}
			}
			for _, dtype := range []mlx.DType{mlx.DTypeBFloat16, mlx.DTypeFloat32} {
				residual := draftTestValues(rng, 2.5, 1, 8, 512).AsType(dtype)
				update := draftTestValues(rng, 0.7, 1, 8, 512)
				want := residual.AsType(mlx.DTypeBFloat16).AsType(mlx.DTypeFloat32).Add(update.AsType(mlx.DTypeFloat32))
				same(fmt.Sprintf("residual sum from %v", dtype), draftResidualSum(residual, update), want)
			}
			state := mlx.Zeros(mlx.DTypeBFloat16, 1, 256)
			for step := range 3 {
				gates := draftTestValues(rng, 3, 1, 768)
				got := compiledGatedState(gates, state)
				wantNext, wantLatent := gatedState(gates, state)
				same(fmt.Sprintf("state %d", step), got[0], wantNext)
				same(fmt.Sprintf("latent %d", step), got[1], wantLatent)
				state = wantNext
			}
			correct := compiledCorrection(30, 0)
			for _, scale := range []float64{4, 40} {
				base := draftTestValues(rng, scale, 1, 4096)
				folded := draftTestValues(rng, scale/4, 1, 4096)
				bias := draftTestValues(rng, scale/8, 4096)
				same(fmt.Sprintf("corrected scores at %v", scale), correct(base, folded, bias), correctScores(base, folded, bias, 30))
			}
		})
	})
}

// draftTestTensors returns a small MTP head under the published tensor names.
func draftTestTensors(rng *rand.Rand, c draftConfig, vocab int) map[string]*mlx.Array {
	norm := func(width int) *mlx.Array {
		return mlx.AddScalar(draftTestValues(rng, 0.1, width).AsType(mlx.DTypeFloat32), 1).AsType(mlx.DTypeBFloat16)
	}
	t := map[string]*mlx.Array{}
	p := "draft."
	t[p+"fuse_weight"] = draftTestValues(rng, 1, c.NumLayers+1, len(c.Taps))
	t[p+"mask_embed"] = draftTestValues(rng, 1, c.Width)
	t[p+"norm.weight"] = norm(c.Width)
	m := p + "markov_head."
	t[m+"dflare_norm.weight"] = norm(c.Width)
	t[m+"markov_w1.weight"] = draftTestValues(rng, 0.5, vocab, c.StateWidth)
	t[m+"rnn_proj.weight"] = draftTestValues(rng, 0.05, 3*c.StateWidth, 2*(c.StateWidth+c.Width))
	t[m+"rnn_proj.bias"] = draftTestValues(rng, 0.1, 3*c.StateWidth)
	t[m+"hidden_out.weight"] = draftTestValues(rng, 0.1, c.Width, c.StateWidth)
	t[m+"hidden_out.bias"] = draftTestValues(rng, 0.1, c.Width)
	kv := c.KVHeads * c.HeadDim
	for i := range c.NumLayers {
		l := fmt.Sprintf("%slayers.%d.", p, i)
		a := l + "self_attn."
		t[l+"input_layernorm.weight"] = norm(c.Width)
		t[l+"post_attention_layernorm.weight"] = norm(c.Width)
		t[l+"dflare_norm.weight"] = norm(c.Width)
		t[a+"q_norm.weight"] = norm(c.HeadDim)
		t[a+"k_norm.weight"] = norm(c.HeadDim)
		t[a+"k_norm_ctx.weight"] = norm(c.HeadDim)
		t[a+"q_proj.weight"] = draftTestValues(rng, 0.08, c.Heads*c.HeadDim, c.Width)
		t[a+"k_proj.weight"] = draftTestValues(rng, 0.08, kv, c.Width)
		t[a+"v_proj.weight"] = draftTestValues(rng, 0.08, kv, c.Width)
		t[a+"k_proj_ctx.weight"] = draftTestValues(rng, 0.08, kv, c.Width)
		t[a+"v_proj_ctx.weight"] = draftTestValues(rng, 0.08, kv, c.Width)
		t[a+"o_proj.weight"] = draftTestValues(rng, 0.08, c.Width, c.Heads*c.HeadDim)
		t[l+"mlp.gate_proj.weight"] = draftTestValues(rng, 0.08, c.FFWidth, c.Width)
		t[l+"mlp.up_proj.weight"] = draftTestValues(rng, 0.08, c.FFWidth, c.Width)
		t[l+"mlp.down_proj.weight"] = draftTestValues(rng, 0.06, c.Width, c.FFWidth)
	}
	return t
}

// A head loaded with the shipped weight widths proposes from scores close to
// the BF16 head's, and every weight it reads is an exported array that the
// runner collects and keeps alive. The 4-bit vocabulary head alone moves the
// scores several times further than the 8-bit projections, so the BF16
// reference reads the same 4-bit vocabulary values: the scores then measure
// the 8-bit drift and the quantized head's product.
func TestDraftQuantizedHeadTracksBF16(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() {
			t.Skip("the draft quantizes its weights on Metal only")
		}
		mlx.Scoped(func() {
			rng := rand.New(rand.NewPCG(29, 3))
			const vocab = 1024
			c := draftConfig{
				Width: 256, NumLayers: 2, Heads: 4, KVHeads: 2, HeadDim: 64, FFWidth: 512,
				LocalLayers: []int{0}, Window: 16, RopeTheta: 1e6, Epsilon: 1e-6, Block: 8,
				StateWidth: 64, HeadType: "rnn_hidden", Taps: []int{1, 2},
			}
			target := &targetModel{
				config:    textConfig{HiddenSize: c.Width, VocabSize: vocab, UnpaddedVocab: vocab, NormEps: 1e-6, LogitCap: 30},
				Embed:     nn.NewEmbedding(draftTestValues(rng, 1, vocab, c.Width)),
				EmbedNorm: mlx.AddScalar(draftTestValues(rng, 0.1, c.Width).AsType(mlx.DTypeFloat32), 1),
				Output:    nn.NewLinear(draftTestValues(rng, 0.1, vocab, c.Width), nil),
			}
			tensors := draftTestTensors(rng, c, vocab)
			load := func(bits, vocabularyBits int) *draft {
				d := &draft{config: c, target: target, prefix: "draft."}
				if err := d.loadWeights(tensors, bits, vocabularyBits); err != nil {
					t.Fatal(err)
				}
				mlx.Eval(mlx.Collect(d)...)
				return d
			}
			quantized, reference := load(draftBits, draftVocabularyBits), load(0, 0)
			vocabulary := quantized.Vocabulary
			if vocabulary == nil || vocabulary.Scales == nil || vocabulary.bits != draftVocabularyBits ||
				quantized.Folded.Scales == nil || quantized.Folded.bits != draftBits || quantized.Layers[0].GateUp.bits != draftBits {
				t.Fatal("the draft did not load the shipped weight widths")
			}
			if reference.Vocabulary != nil {
				t.Fatal("the BF16 draft copied the target's vocabulary head")
			}
			reference.Vocabulary = newDraftProjection(0, mlx.Dequantize(vocabulary.Weight, vocabulary.Scales, vocabulary.Biases, draftGroupSize, vocabulary.bits, "affine", nil))
			collected := mlx.Collect(quantized)
			for _, p := range []*draftProjection{quantized.Vocabulary, quantized.Folded, quantized.Layers[1].QKV, quantized.Layers[1].ContextKV, quantized.Layers[1].Down} {
				for _, a := range []*mlx.Array{p.Weight, p.Scales, p.Biases} {
					if !slices.Contains(collected, a) {
						t.Fatal("a quantized draft weight is not collected")
					}
				}
			}

			proposal := func(d *draft) (hidden, scores *mlx.Array) {
				states := d.NewCaches()
				defer func() {
					for _, s := range states {
						s.Free()
					}
				}()
				width := len(c.Taps) * c.Width
				d.Forward(&batch.Batch{SeqOffsets: []int32{0}, Hidden: draftTestValues(rand.New(rand.NewPCG(1, 1)), 1, 1, 40, width)}, nil, cache.Layers(states))
				b := &batch.Batch{
					InputIDs: mlx.FromValues([]int32{17, 0, 0, 0, 0, 0, 0, 0}, 1, 8), SeqOffsets: []int32{40}, SeqQueryLens: []int32{8},
					Hidden: draftTestValues(rand.New(rand.NewPCG(2, 2)), 1, 1, 3, width),
				}
				h, _ := d.Forward(b, nil, cache.Layers(states))
				rows := h.Slice(mlx.Slice(), mlx.Slice(1, 8), mlx.Slice())
				logits := d.Unembed(rows)
				i := 0
				d.Select(rows.Slice(mlx.Slice(), mlx.Slice(0, 3), mlx.Slice()), logits.Slice(mlx.Slice(), mlx.Slice(0, 3), mlx.Slice()), mlx.FromValues([]int32{17}, 1),
					func(ids, s *mlx.Array) (*mlx.Array, *mlx.Array) {
						// Feed both heads the same proposals so their scores stay comparable.
						if i == 0 {
							scores = s
						} else {
							scores = mlx.Concatenate([]*mlx.Array{scores, s}, 0)
						}
						i++
						return mlx.FromValues([]int32{int32(100 * i)}, 1), s
					})
				mlx.Eval(rows, scores)
				return rows, scores
			}
			// Compare whole score rows; the proposal restriction masks a tail that
			// differs between the heads.
			for _, d := range []*draft{quantized, reference} {
				d.correct = compiledCorrection(d.target.config.LogitCap, 0)
			}
			gotHidden, gotScores := proposal(quantized)
			wantHidden, wantScores := proposal(reference)
			requireDrift(t, "block hidden", gotHidden, wantHidden, 3e-2)
			requireDrift(t, "proposal scores", gotScores, wantScores, 3e-2)
		})
	})
}
