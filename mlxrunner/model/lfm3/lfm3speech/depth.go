package lfm3speech

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// TensorNames locates the speech-output tensors in a checkpoint.
type TensorNames struct {
	CodeEmbedding string // full weight name
	DepthLinear   string
	MaskEmbedding string
	Trunk         string // prefix ending before "layers."
	Codebooks     string // prefix ending before "<k>.embedding.weight"
}

var (
	// OmniNames is the LFM3-Omni layout, which the Audio reference renames to.
	OmniNames = TensorNames{
		CodeEmbedding: "model.codebook_embedding.weight",
		DepthLinear:   "model.depth_decoder.depth_linear",
		MaskEmbedding: "model.depth_decoder.mask_embedding",
		Trunk:         "model.depth_decoder.model.",
		Codebooks:     "model.depth_decoder.depth_embeddings.",
	}
	// AudioNames is the LFM3-Audio training-export layout.
	AudioNames = TensorNames{
		CodeEmbedding: "codebook_embedding.weight",
		DepthLinear:   "depth_linear",
		MaskEmbedding: "mask_embedding",
		Trunk:         "depthformer.model.",
		Codebooks:     "depth_embeddings.",
	}
)

// DepthDecoder predicts the codes of one speech frame from the backbone's
// final-norm hidden state. Each codebook is a slot of a short sequence that a
// bidirectional LFM3 trunk reads; MaskGIT passes reveal the codebooks in
// groups, each pass scoring those still hidden. Array fields are exported so
// mlx.Collect keeps them past the load scope.
type DepthDecoder struct {
	cfg    *Config
	Linear *nn.Linear
	Mask   *mlx.Array // [hidden]
	Layers []*DepthLayer
	Norm   *nn.RMSNorm
	// Codes stacks every codebook's embedding: [codebooks*size, hidden].
	Codes *mlx.Array
	// Heads is the tied output projection, [codebooks, hidden, size].
	Heads *mlx.Array
	// HeadNorms are float32 weights [codebooks, 1, hidden].
	HeadNorms *mlx.Array
	// Cos and Sin are the slot RoPE tables [codebooks, head_dim], rounded
	// through the activation dtype as the reference rounds them.
	Cos, Sin *mlx.Array
}

type DepthLayer struct {
	AttnNorm, FFNNorm *nn.RMSNorm
	Attn              *Attention
	MLP               *MLP
}

// LoadDepthDecoder builds the depth decoder from unquantized weights.
func LoadDepthDecoder(tensors map[string]*mlx.Array, names TensorNames, cfg *Config) (*DepthDecoder, error) {
	dc := cfg.Depth
	K, size, hidden := cfg.NumCodebooks, cfg.CodebookSize, int(dc.HiddenSize)
	d := &DepthDecoder{cfg: cfg}
	var err error
	if d.Linear, err = dense(tensors, names.DepthLinear, K*hidden, -1, true); err != nil {
		return nil, err
	}
	if d.Mask = tensors[names.MaskEmbedding]; d.Mask == nil || d.Mask.NumDims() != 1 || d.Mask.Dim(0) != hidden {
		return nil, fmt.Errorf("missing or misshapen %s", names.MaskEmbedding)
	}
	if d.Norm, err = rmsNorm(tensors, names.Trunk+"embedding_norm.weight", hidden, dc.NormEps); err != nil {
		return nil, err
	}
	q, kv := int(dc.NumHeads*dc.HeadDim), int(dc.NumKVHeads*dc.HeadDim)
	for i := range int(dc.NumHiddenLayers) {
		p := fmt.Sprintf("%slayers.%d.", names.Trunk, i)
		l := &DepthLayer{Attn: &Attention{}, MLP: &MLP{}}
		if tensors[p+"conv.conv.weight"] != nil || tensors[p+"post_operator_norm.weight"] != nil {
			return nil, fmt.Errorf("depth layer %d is not a plain attention layer", i)
		}
		linears := []struct {
			dst     *nn.LinearLayer
			name    string
			out, in int
		}{
			{&l.Attn.QProj, "self_attn.q_proj", q, hidden},
			{&l.Attn.KProj, "self_attn.k_proj", kv, hidden},
			{&l.Attn.VProj, "self_attn.v_proj", kv, hidden},
			{&l.Attn.OProj, "self_attn.o_proj", hidden, q},
			{&l.Attn.GateProj, "self_attn.gate_proj", int(dc.NumHeads), hidden},
			{&l.MLP.W1, "feed_forward.w1", int(dc.IntermediateSize), hidden},
			{&l.MLP.W3, "feed_forward.w3", int(dc.IntermediateSize), hidden},
			{&l.MLP.W2, "feed_forward.w2", hidden, int(dc.IntermediateSize)},
		}
		for _, lin := range linears {
			layer, err := dense(tensors, p+lin.name, lin.out, lin.in, false)
			if err != nil {
				return nil, err
			}
			*lin.dst = layer
		}
		if l.Attn.QNorm, err = rmsNorm(tensors, p+"self_attn.q_norm.weight", int(dc.HeadDim), dc.NormEps); err != nil {
			return nil, err
		}
		if l.Attn.KNorm, err = rmsNorm(tensors, p+"self_attn.k_norm.weight", int(dc.HeadDim), dc.NormEps); err != nil {
			return nil, err
		}
		if l.AttnNorm, err = rmsNorm(tensors, p+"operator_norm.weight", hidden, dc.NormEps); err != nil {
			return nil, err
		}
		if l.FFNNorm, err = rmsNorm(tensors, p+"ffn_norm.weight", hidden, dc.NormEps); err != nil {
			return nil, err
		}
		d.Layers = append(d.Layers, l)
	}
	if tensors[fmt.Sprintf("%slayers.%d.operator_norm.weight", names.Trunk, dc.NumHiddenLayers)] != nil {
		return nil, fmt.Errorf("depth trunk has more than %d layers", dc.NumHiddenLayers)
	}

	tables := make([]*mlx.Array, K)
	norms := make([]*mlx.Array, K)
	for k := range K {
		name := fmt.Sprintf("%s%d.embedding.weight", names.Codebooks, k)
		w := tensors[name]
		if w == nil || w.NumDims() != 2 || w.Dim(0) != size || w.Dim(1) != hidden {
			return nil, fmt.Errorf("missing or misshapen %s", name)
		}
		if err := requireDense(tensors, name); err != nil {
			return nil, err
		}
		tables[k] = w
		n, err := rmsNorm(tensors, fmt.Sprintf("%s%d.embedding_norm.weight", names.Codebooks, k), hidden, dc.NormEps)
		if err != nil {
			return nil, err
		}
		norms[k] = n.Weight.AsType(mlx.DTypeFloat32)
	}
	d.Codes = mlx.Concatenate(tables, 0)
	d.Heads = mlx.Transpose(mlx.Reshape(d.Codes, int32(K), int32(size), int32(hidden)), 0, 2, 1)
	d.HeadNorms = mlx.Reshape(mlx.Stack(norms, 0), int32(K), 1, int32(hidden))

	// HF RoPE over all head dims: angle(p, i) = p * theta^(-2i/d), the two
	// halves sharing one frequency; tables are rounded through the dtype.
	half := int(dc.HeadDim) / 2
	cos := make([]float32, K*int(dc.HeadDim))
	sin := make([]float32, K*int(dc.HeadDim))
	for p := range K {
		for i := range half {
			invFreq := float32(1 / math.Pow(float64(dc.RopeTheta), float64(2*i)/float64(dc.HeadDim)))
			angle := float64(float32(p) * invFreq)
			for _, j := range []int{i, i + half} {
				cos[p*int(dc.HeadDim)+j] = float32(math.Cos(angle))
				sin[p*int(dc.HeadDim)+j] = float32(math.Sin(angle))
			}
		}
	}
	dtype := d.Linear.Weight.DType()
	d.Cos = mlx.FromValues(cos, K, int(dc.HeadDim)).AsType(dtype).AsType(mlx.DTypeFloat32)
	d.Sin = mlx.FromValues(sin, K, int(dc.HeadDim)).AsType(dtype).AsType(mlx.DTypeFloat32)
	return d, nil
}

// project maps a hidden state ([hidden], [1, hidden] or [1, 1, hidden]) to
// the per-codebook slot inputs [1, codebooks, depth hidden].
func (d *DepthDecoder) project(hidden *mlx.Array) *mlx.Array {
	h := mlx.Reshape(hidden, 1, -1)
	return mlx.Reshape(d.Linear.Forward(h), 1, int32(d.cfg.NumCodebooks), d.cfg.Depth.HiddenSize)
}

// logits scores every codebook given the revealed codes; masked codebooks
// see the mask embedding instead of their code. Returns [codebooks, size].
func (d *DepthDecoder) logits(slots *mlx.Array, codes []int32, masked []bool) *mlx.Array {
	K := d.cfg.NumCodebooks
	rows := make([]int32, K)
	for k, c := range codes {
		if !masked[k] {
			rows[k] = int32(k*d.cfg.CodebookSize) + c
		}
	}
	embedded := mlx.Take(d.Codes, mlx.FromValues(rows, K), 0)
	inputs := mlx.Where(mlx.FromValues(masked, K, 1), d.Mask.AsType(embedded.DType()), embedded)
	x := mlx.Add(slots, mlx.ExpandDims(inputs, 0))
	for _, l := range d.Layers {
		x = mlx.Add(x, l.attention(l.AttnNorm.Forward(x, 0), d))
		x = mlx.Add(x, l.MLP.Forward(l.FFNNorm.Forward(x, 0)))
	}
	x = d.Norm.Forward(x, 0)

	// Head norms scale in float32 and round once, unlike the trunk's.
	h := mlx.Reshape(x.AsType(mlx.DTypeFloat32), int32(K), 1, d.cfg.Depth.HiddenSize)
	variance := mlx.AddScalar(mlx.Mean(mlx.Mul(h, h), -1, true), d.cfg.Depth.NormEps)
	h = mlx.Mul(mlx.Mul(h, mlx.RSqrt(variance)), d.HeadNorms).AsType(x.DType())
	return mlx.Reshape(mlx.Matmul(h, d.Heads), int32(K), int32(d.cfg.CodebookSize))
}

// attention is bidirectional over the codebook slots.
func (l *DepthLayer) attention(x *mlx.Array, d *DepthDecoder) *mlx.Array {
	dc := d.cfg.Depth
	L := int32(d.cfg.NumCodebooks)
	heads := func(proj nn.LinearLayer, n int32) *mlx.Array {
		return mlx.Transpose(mlx.Reshape(proj.Forward(x), 1, L, n, dc.HeadDim), 0, 2, 1, 3)
	}
	q := d.rope(l.Attn.QNorm.Forward(heads(l.Attn.QProj, dc.NumHeads), 0))
	k := d.rope(l.Attn.KNorm.Forward(heads(l.Attn.KProj, dc.NumKVHeads), 0))
	v := heads(l.Attn.VProj, dc.NumKVHeads)
	out := mlx.FastScaledDotProductAttention(q, k, v, float32(1/math.Sqrt(float64(dc.HeadDim))), mlx.FastSDPANoWindow, "", nil)
	out = mlx.Transpose(out, 0, 2, 1, 3)
	gate := mlx.Sigmoid(l.Attn.GateProj.Forward(x))
	out = mlx.Mul(out, mlx.ExpandDims(gate, -1))
	return l.Attn.OProj.Forward(mlx.Reshape(out, 1, L, dc.NumHeads*dc.HeadDim))
}

// rope rotates x [1, heads, slots, head_dim] in float32 with the
// non-interleaved half layout.
func (d *DepthDecoder) rope(x *mlx.Array) *mlx.Array {
	half := d.cfg.Depth.HeadDim / 2
	f := x.AsType(mlx.DTypeFloat32)
	B, H, L := int32(f.Dim(0)), int32(f.Dim(1)), int32(f.Dim(2))
	lo := mlx.SliceStartStop(f, []int32{0, 0, 0, 0}, []int32{B, H, L, half})
	hi := mlx.SliceStartStop(f, []int32{0, 0, 0, half}, []int32{B, H, L, 2 * half})
	rotated := mlx.Concatenate([]*mlx.Array{mlx.Neg(hi), lo}, 3)
	return mlx.Add(mlx.Mul(f, d.Cos), mlx.Mul(rotated, d.Sin)).AsType(x.DType())
}

// Logits runs one depth pass; it exists for parity tests. codes of masked
// codebooks are ignored.
func (d *DepthDecoder) Logits(hidden *mlx.Array, codes []int32, masked []bool) *mlx.Array {
	return d.logits(d.project(hidden), codes, masked)
}

// DecodeFrame samples one speech frame from the backbone hidden state that
// precedes it. When any codebook draws the EOS code it returns an all-EOS
// frame and eos; that frame is still fed back to the backbone, but it is
// neither audio nor penalty history.
func (d *DepthDecoder) DecodeFrame(hidden *mlx.Array, s *FrameSampler) (frame []int32, eos bool) {
	K, size := d.cfg.NumCodebooks, d.cfg.CodebookSize
	frame = make([]int32, K)
	masked := make([]bool, K)
	for k := range masked {
		masked[k] = true
	}
	order := s.order(K)
	mlx.Scoped(func() {
		slots := d.project(hidden)
		start := 0
		for _, n := range d.cfg.Schedule {
			pick := order[start : start+n]
			start += n
			rows := make([]int32, n)
			for i, k := range pick {
				rows[i] = int32(k)
			}
			logits := mlx.Take(d.logits(slots, frame, masked), mlx.FromValues(rows, n), 0).AsType(mlx.DTypeFloat32).Floats()
			for i, k := range pick {
				frame[k] = s.sample(logits[i*size:(i+1)*size], k)
			}
			for _, k := range pick {
				masked[k] = false
			}
		}
	})
	for _, c := range frame {
		if c == d.cfg.EOSCode {
			for k := range frame {
				frame[k] = d.cfg.EOSCode
			}
			return frame, true
		}
	}
	s.remember(frame)
	return frame, false
}

// dense loads an unquantized linear layer, checking its shape; in < 0 skips
// the input check.
func dense(tensors map[string]*mlx.Array, prefix string, out, in int, bias bool) (*nn.Linear, error) {
	w := tensors[prefix+".weight"]
	if w == nil {
		return nil, fmt.Errorf("missing %s.weight", prefix)
	}
	if err := requireDense(tensors, prefix+".weight"); err != nil {
		return nil, err
	}
	if w.NumDims() != 2 || w.Dim(0) != out || (in >= 0 && w.Dim(1) != in) {
		return nil, fmt.Errorf("%s.weight has shape %v, want [%d, %d]", prefix, w.Dims(), out, in)
	}
	b := tensors[prefix+".bias"]
	if bias != (b != nil) || (b != nil && (b.NumDims() != 1 || b.Dim(0) != out)) {
		return nil, fmt.Errorf("%s.bias presence or shape is unexpected", prefix)
	}
	return nn.NewLinear(w, b), nil
}

func rmsNorm(tensors map[string]*mlx.Array, name string, dim int, eps float32) (*nn.RMSNorm, error) {
	w := tensors[name]
	if w == nil || w.NumDims() != 1 || w.Dim(0) != dim {
		return nil, fmt.Errorf("missing or misshapen %s", name)
	}
	return nn.NewRMSNorm(w, eps), nil
}
