package laya

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

type Model struct {
	Embeddings    *mlx.Array
	TypeEmbedding *mlx.Array
	EmbeddingNorm *nn.LayerNorm
	FinalNorm     *nn.LayerNorm
	Layers        []encoderLayer
	Head          []headLayer
	ScoreNorm     *nn.LayerNorm
	ScoreIn       *nn.Linear
	ScoreOut      *nn.Linear

	encoder        encoderConfig
	config         config
	tok            *tokenizer.Tokenizer
	cls, sep, mask int32
	maskToken      string
}

type encoderLayer struct {
	AttentionNorm, MLPNorm *nn.LayerNorm
	QKV, Out, In, Down     *nn.Linear
}

type headLayer struct {
	Norm1, Norm2       *nn.LayerNorm
	QKV, Out, In, Down *nn.Linear
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }
func (m *Model) MaxContextLength() int           { return m.config.MaxLen }
func (m *Model) NewCaches() []cache.Cache        { return nil }

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	var loadErr error
	weight := func(name string, shape ...int) *mlx.Array {
		w := tensors[name]
		if w == nil {
			if loadErr == nil {
				loadErr = fmt.Errorf("missing Laya weight %s", name)
			}
			return nil
		}
		valid := w.NumDims() == len(shape)
		if valid {
			for i, n := range shape {
				valid = valid && w.Dim(i) == n
			}
		}
		if !valid {
			if loadErr == nil {
				loadErr = fmt.Errorf("invalid shape for Laya weight %s: want %v", name, shape)
			}
			return nil
		}
		if w.DType() != mlx.DTypeFloat16 && w.DType() != mlx.DTypeBFloat16 && w.DType() != mlx.DTypeFloat32 {
			if loadErr == nil {
				loadErr = fmt.Errorf("Laya requires unquantized floating-point weights: %s", name)
			}
			return nil
		}
		return w
	}
	d := m.encoder.HiddenSize
	norm := func(name string, bias bool, eps float32) *nn.LayerNorm {
		l := &nn.LayerNorm{Weight: weight(name+".weight", d), Eps: eps}
		if bias {
			l.Bias = weight(name+".bias", d)
		}
		return l
	}
	linear := func(name string, out, in int, bias bool) *nn.Linear {
		l := &nn.Linear{Weight: weight(name+".weight", out, in)}
		if bias {
			l.Bias = weight(name+".bias", out)
		}
		return l
	}
	m.Embeddings = weight("encoder.embeddings.tok_embeddings.weight", m.encoder.VocabSize, d)
	m.TypeEmbedding = weight("type_emb.weight", 3, d)
	m.EmbeddingNorm = norm("encoder.embeddings.norm", false, m.encoder.NormEps)
	m.FinalNorm = norm("encoder.final_norm", false, m.encoder.NormEps)
	m.Layers = make([]encoderLayer, m.encoder.Layers)
	for i := range m.Layers {
		p := fmt.Sprintf("encoder.layers.%d", i)
		l := &m.Layers[i]
		if i != 0 {
			l.AttentionNorm = norm(p+".attn_norm", false, m.encoder.NormEps)
		}
		l.MLPNorm = norm(p+".mlp_norm", false, m.encoder.NormEps)
		l.QKV, l.Out = linear(p+".attn.Wqkv", 3*d, d, false), linear(p+".attn.Wo", d, d, false)
		l.In = linear(p+".mlp.Wi", 2*m.encoder.IntermediateSize, d, false)
		l.Down = linear(p+".mlp.Wo", d, m.encoder.IntermediateSize, false)
	}
	m.Head = make([]headLayer, m.config.HeadLayers)
	for i := range m.Head {
		p := fmt.Sprintf("head.layers.%d", i)
		l := &m.Head[i]
		l.Norm1, l.Norm2 = norm(p+".norm1", true, 1e-5), norm(p+".norm2", true, 1e-5)
		l.QKV = &nn.Linear{Weight: weight(p+".self_attn.in_proj_weight", 3*d, d), Bias: weight(p+".self_attn.in_proj_bias", 3*d)}
		l.Out = linear(p+".self_attn.out_proj", d, d, true)
		l.In, l.Down = linear(p+".linear1", 4*d, d, true), linear(p+".linear2", d, 4*d, true)
	}
	m.ScoreNorm = norm("scorer.0", true, 1e-5)
	m.ScoreIn, m.ScoreOut = linear("scorer.1", d, d, true), linear("scorer.3", 1, d, true)
	return loadErr
}

// attention is bidirectional. Only the encoder's local layers use a window;
// the learned decision head attends to every token without rotary positions.
func attention(b *batch.Batch, qkv *mlx.Array, heads int, theta float32, mask nn.AttentionMask) *mlx.Array {
	B, n, width := qkv.Dim(0), qkv.Dim(1), qkv.Dim(2)/3
	part := func(i int) *mlx.Array {
		return qkv.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(i*width, (i+1)*width)).Reshape(B, n, heads, width/heads).Transpose(0, 2, 1, 3)
	}
	q, k, v := part(0), part(1), part(2)
	if theta != 0 {
		offset := mlx.FromValues(b.SeqOffsets, B)
		q = mlx.RoPEWithBase(q, width/heads, false, theta, 1, offset)
		k = mlx.RoPEWithBase(k, width/heads, false, theta, 1, offset)
	}
	return nn.ScaledDotProductAttention(b, q, float32(1/math.Sqrt(float64(width/heads))), nn.WithKV(k, v, b.SeqQueryLens), nn.WithMask(mask)).Transpose(0, 2, 1, 3).Reshape(B, n, width)
}

func (m *Model) Forward(b *batch.Batch, _ []cache.Cache) (*mlx.Array, *mlx.Array) {
	h := m.encoderForward(b)
	// The question type is model-owned row metadata. A missing layout denotes
	// the choice type for load-time probes.
	types := make([]int32, b.InputIDs.Dim(0))
	for i, layout := range b.Layout {
		if layout != nil {
			types[i] = layout.(int32)
		}
	}
	h = h.Add(m.TypeEmbedding.TakeAxis(mlx.FromValues(types, len(types)), 0).ExpandDims(1))
	for _, l := range m.Head {
		h = l.forward(b, h, max(1, m.encoder.HiddenSize/64))
	}
	return h, nil
}

func (m *Model) encoderForward(b *batch.Batch) *mlx.Array {
	h := m.EmbeddingNorm.Forward(m.Embeddings.TakeAxis(b.InputIDs, 0))
	// Build the quadratic mask on the device; host input remains linear in n.
	positions := mlx.Arange(0, float64(b.InputIDs.Dim(1)), 1, mlx.DTypeInt32)
	local := positions.ExpandDims(0).Subtract(positions.ExpandDims(1)).Abs().LessEqual(mlx.NewScalarArray(float32(m.encoder.LocalAttention / 2)))
	localMask := nn.ArrayMask(mlx.Where(local, mlx.FromValue(float32(0)), mlx.FromValue(float32(math.Inf(-1)))))
	for i, l := range m.Layers {
		theta, mask := m.encoder.GlobalTheta, nn.AttentionMask{}
		if i%m.encoder.GlobalEvery != 0 {
			theta, mask = m.encoder.LocalTheta, localMask
		}
		h = l.forward(b, h, m.encoder.Heads, theta, mask)
	}
	return m.FinalNorm.Forward(h)
}

func (l encoderLayer) forward(b *batch.Batch, h *mlx.Array, heads int, theta float32, mask nn.AttentionMask) *mlx.Array {
	x := h
	if l.AttentionNorm != nil {
		x = l.AttentionNorm.Forward(x)
	}
	h = h.Add(l.Out.Forward(attention(b, l.QKV.Forward(x), heads, theta, mask)))
	up := l.In.Forward(l.MLPNorm.Forward(h))
	d := up.Dim(2) / 2
	activated := mlx.GELU(up.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, d))).Multiply(up.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(d, 2*d)))
	return h.Add(l.Down.Forward(activated))
}

func (l headLayer) forward(b *batch.Batch, h *mlx.Array, heads int) *mlx.Array {
	h = h.Add(l.Out.Forward(attention(b, l.QKV.Forward(l.Norm1.Forward(h)), heads, 0, nn.AttentionMask{})))
	// PyTorch TransformerEncoderLayer defaults to ReLU, not GELU.
	return h.Add(l.Down.Forward(mlx.ReLU(l.In.Forward(l.Norm2.Forward(h)))))
}

func (m *Model) Unembed(h *mlx.Array) *mlx.Array {
	return m.ScoreOut.Forward(mlx.GELU(m.ScoreIn.Forward(m.ScoreNorm.Forward(h))))
}
