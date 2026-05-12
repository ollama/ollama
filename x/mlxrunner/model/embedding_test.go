package model

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

func skipIfNoMLX(t *testing.T) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
}

func TestMakeEmbeddingLayerDense(t *testing.T) {
	skipIfNoMLX(t)

	weight := mlx.FromValues([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}, 2, 4).AsType(mlx.DTypeBFloat16)

	emb := MakeEmbeddingLayer(map[string]*mlx.Array{
		"model.embed_tokens.weight": weight,
	}, "model.embed_tokens", 0, 0, "", nil)

	dense, ok := emb.(*nn.Embedding)
	if !ok {
		t.Fatalf("embedding type = %T, want *nn.Embedding", emb)
	}
	if dense.Weight.DType() != mlx.DTypeBFloat16 {
		t.Fatalf("embedding dtype = %v, want %v", dense.Weight.DType(), mlx.DTypeBFloat16)
	}
	if _, ok := emb.AsLinear().(*nn.Linear); !ok {
		t.Fatalf("AsLinear type = %T, want *nn.Linear", emb.AsLinear())
	}
}

func TestMakeEmbeddingLayerQuantized(t *testing.T) {
	skipIfNoMLX(t)

	denseWeight := mlx.FromValues(func() []float32 {
		out := make([]float32, 2*64)
		for i := range out {
			out[i] = float32(i%17) / 8
		}
		return out
	}(), 2, 64).AsType(mlx.DTypeBFloat16)

	qw, scales, qbiases := mlx.Quantize(denseWeight, 64, 4, "affine")
	mlx.Eval(qw, scales, qbiases)

	emb := MakeEmbeddingLayer(map[string]*mlx.Array{
		"model.embed_tokens.weight":       qw,
		"model.embed_tokens.weight_scale": scales,
		"model.embed_tokens.weight_qbias": qbiases,
	}, "model.embed_tokens", 64, 4, "affine", nil)

	qemb, ok := emb.(*nn.QuantizedEmbedding)
	if !ok {
		t.Fatalf("embedding type = %T, want *nn.QuantizedEmbedding", emb)
	}
	if qemb.GroupSize != 64 || qemb.Bits != 4 || qemb.Mode != "affine" {
		t.Fatalf("quant params = (%d, %d, %q), want (64, 4, %q)", qemb.GroupSize, qemb.Bits, qemb.Mode, "affine")
	}

	indices := mlx.FromValues([]int32{1, 0}, 2)
	out := emb.Forward(indices)
	mlx.Eval(out)
	if dims := out.Dims(); len(dims) != 2 || dims[0] != 2 || dims[1] != 64 {
		t.Fatalf("embedding output dims = %v, want [2 64]", dims)
	}
	if _, ok := emb.AsLinear().(*nn.QuantizedLinear); !ok {
		t.Fatalf("AsLinear type = %T, want *nn.QuantizedLinear", emb.AsLinear())
	}
}
