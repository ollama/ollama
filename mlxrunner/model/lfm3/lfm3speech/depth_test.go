package lfm3speech

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

// tinyDepth builds a random depth decoder: 3 codebooks of 5 codes, two
// bidirectional layers, reading a 6-wide backbone state.
func tinyDepth(t *mlxtest.T) (*DepthDecoder, *CodeEmbedding, map[string]*mlx.Array) {
	cfg := &Config{
		NumCodebooks: 3, CodebookSize: 5, EOSCode: 4, Schedule: []int{1, 2},
		InterleavedText: 6, InterleavedAudio: 12,
		Depth: DepthConfig{
			HiddenSize: 8, IntermediateSize: 12, NumHiddenLayers: 2, NumHeads: 2, NumKVHeads: 1, HeadDim: 4,
			NormEps: 1e-5, MaxPositions: 3, LayerTypes: []string{"full_attention", "full_attention"}, RopeTheta: 1e6,
		},
	}
	if err := cfg.validate(); err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(11, 12))
	n := OmniNames
	ts := map[string]*mlx.Array{}
	put := func(name string, scale float64, shape ...int) { ts[name] = randArray(rng, scale, shape...) }
	ones := func(name string, size int) {
		v := make([]float32, size)
		for i := range v {
			v[i] = float32(1 + 0.1*rng.NormFloat64())
		}
		ts[name] = mlx.FromValues(v, size)
	}
	put(n.CodeEmbedding, 1, 15, 6)
	put(n.DepthLinear+".weight", 0.5, 24, 6)
	put(n.DepthLinear+".bias", 0.1, 24)
	put(n.MaskEmbedding, 0.5, 8)
	ones(n.Trunk+"embedding_norm.weight", 8)
	for i := range 2 {
		p := fmt.Sprintf("%slayers.%d.", n.Trunk, i)
		ones(p+"operator_norm.weight", 8)
		ones(p+"ffn_norm.weight", 8)
		put(p+"self_attn.q_proj.weight", 0.4, 8, 8)
		put(p+"self_attn.k_proj.weight", 0.4, 4, 8)
		put(p+"self_attn.v_proj.weight", 0.4, 4, 8)
		put(p+"self_attn.o_proj.weight", 0.4, 8, 8)
		put(p+"self_attn.gate_proj.weight", 0.4, 2, 8)
		ones(p+"self_attn.q_norm.weight", 4)
		ones(p+"self_attn.k_norm.weight", 4)
		put(p+"feed_forward.w1.weight", 0.4, 12, 8)
		put(p+"feed_forward.w3.weight", 0.4, 12, 8)
		put(p+"feed_forward.w2.weight", 0.4, 8, 12)
	}
	for k := range 3 {
		put(fmt.Sprintf("%s%d.embedding.weight", n.Codebooks, k), 1, 5, 8)
		ones(fmt.Sprintf("%s%d.embedding_norm.weight", n.Codebooks, k), 8)
	}
	d, err := LoadDepthDecoder(ts, n, cfg)
	if err != nil {
		t.Fatal(err)
	}
	e, err := LoadCodeEmbedding(ts, n.CodeEmbedding, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return d, e, ts
}

func floats(a *mlx.Array) []float32 {
	var out []float32
	mlx.Scoped(func() { out = a.AsType(mlx.DTypeFloat32).Floats() })
	return out
}

func maxDiff(a, b []float32) float64 {
	var m float64
	for i := range a {
		m = math.Max(m, math.Abs(float64(a[i]-b[i])))
	}
	return m
}

func TestDepthDecoderMasking(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		d, _, _ := tinyDepth(t)
		h := randArray(rand.New(rand.NewPCG(1, 1)), 1, 6)
		all := []bool{true, true, true}
		// Hidden codebooks contribute only the mask embedding.
		a := floats(d.Logits(h, []int32{0, 1, 2}, all))
		b := floats(d.Logits(h, []int32{3, 3, 0}, all))
		if maxDiff(a, b) != 0 {
			t.Fatal("masked codes changed the logits")
		}
		// Revealing the last slot changes the first slot's logits: the trunk
		// attends both ways.
		c := floats(d.Logits(h, []int32{0, 0, 3}, []bool{true, true, false}))
		if maxDiff(a[:5], c[:5]) < 1e-4 {
			t.Fatal("slot 0 ignores a later slot; attention is not bidirectional")
		}
	})
}

func TestDecodeFrameGreedyFixedOrder(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		d, _, _ := tinyDepth(t)
		h := randArray(rand.New(rand.NewPCG(2, 2)), 1, 1, 1, 6)
		order := []int{2, 0, 1}
		// Replay the schedule [1, 2] by hand with argmax.
		want := make([]int32, 3)
		masked := []bool{true, true, true}
		for _, group := range [][]int{order[:1], order[1:]} {
			logits := floats(d.Logits(h, want, masked))
			for _, k := range group {
				want[k] = int32(argmax(logits[k*5 : (k+1)*5]))
			}
			for _, k := range group {
				masked[k] = false
			}
		}
		// Move EOS to a code the greedy frame does not use.
		for slices.Contains(want, d.cfg.EOSCode) {
			d.cfg.EOSCode--
		}
		s, err := NewFrameSampler(d.cfg, SampleOptions{Order: order, RepetitionWindow: 4}, nil)
		if err != nil {
			t.Fatal(err)
		}
		got, eos := d.DecodeFrame(h, s)
		if eos || !slices.Equal(got, want) {
			t.Fatalf("frame %v eos %v, want %v", got, eos, want)
		}
		if len(s.history) != 1 || !slices.Equal(s.history[0], want) {
			t.Fatalf("history %v", s.history)
		}

		// Make the greedy code of the first revealed codebook the EOS code.
		d.cfg.EOSCode = want[order[0]]
		got, eos = d.DecodeFrame(h, s)
		if !eos || !slices.Equal(got, []int32{want[order[0]], want[order[0]], want[order[0]]}) {
			t.Fatalf("EOS frame %v eos %v", got, eos)
		}
		if len(s.history) != 1 {
			t.Fatal("an EOS frame entered the repetition history")
		}
	})
}

// The runner keeps only what mlx.Collect finds on the model after its load
// scope; everything the components use must survive that.
func TestWeightsSurviveLoadScope(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		var d *DepthDecoder
		var e *CodeEmbedding
		var c *Codec
		kept := mlx.ScopedArrays(func() []*mlx.Array {
			d, e, _ = tinyDepth(t)
			c = tinyCodec(t)
			return mlx.Collect(&struct {
				D *DepthDecoder
				E *CodeEmbedding
				C *Codec
			}{d, e, c})
		})
		mlx.Eval(kept...)
		held := mlx.NewScope()
		held.Attach(kept...)
		defer held.Close()
		mlx.Scoped(func() {
			s, err := NewFrameSampler(d.cfg, SampleOptions{Order: []int{0, 1, 2}}, nil)
			if err != nil {
				t.Fatal(err)
			}
			d.DecodeFrame(mlx.Zeros(mlx.DTypeFloat32, 6), s)
			if _, err := e.Forward([]int32{1, 2, 3}); err != nil {
				t.Fatal(err)
			}
			st := c.NewStream()
			defer st.Close()
			if _, err := st.Push([]int32{1, 2}); err != nil {
				t.Fatal(err)
			}
		})
	})
}

func TestCodeEmbeddingSum(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		_, e, ts := tinyDepth(t)
		table := floats(ts[OmniNames.CodeEmbedding])
		frames := [][]int32{{0, 4, 2}, {3, 1, 4}}
		out, err := e.Forward(frames...)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(out.Dims(), []int{2, 6}) {
			t.Fatalf("embedding shape %v", out.Dims())
		}
		got := floats(out)
		for f, frame := range frames {
			for j := range 6 {
				var want float32
				for k, c := range frame {
					want += table[(k*5+int(c))*6+j]
				}
				if math.Abs(float64(got[f*6+j]-want)) > 1e-5 {
					t.Fatalf("frame %d dim %d = %v, want %v", f, j, got[f*6+j], want)
				}
			}
		}
		if _, err := e.Forward([]int32{0, 5, 0}); err == nil {
			t.Fatal("accepted a code past the codebook")
		}
	})
}
