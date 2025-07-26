package ggml

import (
	"encoding/json"
	"flag"
	"log/slog"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/fast"
	"github.com/ollama/ollama/ml/nn/rope"
)

var args struct {
	model,
	output,
	image string
	tokens,
	layer int
}

func TestMain(m *testing.M) {
	flag.IntVar(&args.tokens, "tokens", 10, "number of tokens to generate")
	flag.StringVar(&args.model, "model", "", "path to model")
	flag.StringVar(&args.output, "output", "", "file for output tensor")
	flag.StringVar(&args.image, "image", "", "path to image for multimodal models")
	flag.IntVar(&args.layer, "layer", math.MaxInt, "number of layers to use for the model, -1 for all layers")
	flag.Parse()

	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))

	os.Exit(m.Run())
}

func setup(tb testing.TB) ml.Backend {
	models := envconfig.Models()

	model, tag, found := strings.Cut(args.model, ":")
	if !found {
		tag = "latest"
	}

	manifest, err := os.Open(filepath.Join(models, "manifests", "registry.ollama.ai", "library", model, tag))
	if err != nil {
		tb.Fatal(err)
	}
	defer manifest.Close()

	var m struct {
		Layers []struct {
			MediaType string `json:"mediaType"`
			Digest    string `json:"digest"`
		} `json:"layers"`
	}

	if err := json.NewDecoder(manifest).Decode(&m); err != nil {
		tb.Fatal(err)
	}

	for _, layer := range m.Layers {
		tb.Log(layer.MediaType, layer.Digest)
		if layer.MediaType == "application/vnd.ollama.image.model" {
			b, err := New(filepath.Join(models, "blobs", strings.ReplaceAll(layer.Digest, ":", "-")), ml.BackendParams{})
			if err != nil {
				tb.Fatal(err)
			}

			return b
		}
	}

	tb.Fatalf("no image model found in %s", filepath.Join(models, "blobs", strings.ReplaceAll(m.Layers[0].Digest, ":", "-")))
	return nil
}

func mul(shape ...int) int {
	m := 1
	for _, s := range shape {
		m *= s
	}
	return m
}

func TestRMSNorm(t *testing.T) {
	b := setup(t)

	t.Run("rmsnorm2d", func(t *testing.T) {
		ctx := b.NewContext()
		defer ctx.Close()

		tt := ctx.Arange(0, 2*2*3, 1, ml.DTypeF32).Reshape(ctx, 2, 2, 3)
		t.Log("tt:", tt.Shape())
		// t.Log(ml.Dump(ctx, a))

		var rms nn.RMSNorm
		tt = rms.Forward(ctx, tt, 1e-5)
		t.Log("tt:", tt.Shape())
		t.Log(ml.Dump(ctx, tt))
	})
}

func TestQKV(t *testing.T) {
	b := setup(t)

	t.Run("qkv", func(t *testing.T) {
		ctx := b.NewContext()
		defer ctx.Close()

		tt := ctx.Arange(0, 4*8, 1, ml.DTypeF32).Reshape(ctx, 8, 4)
		t.Log("tt:", tt.Shape())
		// t.Log(ml.Dump(ctx, a))

		qkv := nn.Linear{
			Weight: ctx.Arange(0, 8*12, 1, ml.DTypeF32).Reshape(ctx, 8, 12),
		}

		tt = qkv.Forward(ctx, tt)
		t.Log("tt:", tt.Shape())
		// t.Log(ml.Dump(ctx, tt))

		tt = tt.View(ctx, 0, 8, tt.Stride(1), tt.Dim(1)).Contiguous(ctx)
		t.Log("tt:", tt.Shape())
		t.Log(ml.Dump(ctx, tt))
	})

	t.Run("qkv", func(t *testing.T) {
		ctx := b.NewContext()
		defer ctx.Close()

		tt := ctx.Arange(0, 4*8, 1, ml.DTypeF32).Reshape(ctx, 8, 4)
		t.Log("tt:", tt.Shape())
		// t.Log(ml.Dump(ctx, a))

		qkv := nn.Linear{
			Weight: ctx.Arange(0, 8*12, 1, ml.DTypeF32).Reshape(ctx, 8, 12),
		}

		tt = qkv.Forward(ctx, tt)
		t.Log("tt:", tt.Shape())
		// t.Log(ml.Dump(ctx, tt))

		tt = tt.View(ctx, 0, 8, tt.Stride(1), tt.Dim(1)).Contiguous(ctx)
		t.Log("tt:", tt.Shape())
		t.Log(ml.Dump(ctx, tt))
	})

	t.Run("q", func(t *testing.T) {
		ctx := b.NewContext()
		defer ctx.Close()

		tt := ctx.Arange(0, 4*8, 1, ml.DTypeF32).Reshape(ctx, 8, 4)
		t.Log("tt:", tt.Shape())

		q := nn.Linear{
			Weight: ctx.Arange(0, 8*8, 1, ml.DTypeF32).Reshape(ctx, 8, 8),
		}

		tt = q.Forward(ctx, tt)
		t.Log("tt:", tt.Shape())
		// t.Log(ml.Dump(ctx, tt))

		tt = tt.View(ctx, 0, 8, tt.Stride(1), tt.Dim(1)).Contiguous(ctx)
		t.Log("tt:", tt.Shape())
		t.Log(ml.Dump(ctx, tt))
	})
}

func TestYaRN(t *testing.T) {
	b := setup(t)

	tokens := 100
	headDim := 64
	// originalContextLength := 4 << 10
	ropeTheta := 150000
	ropeScalingFactor := 32
	// ntkBeta := 32
	// ntkAlpha := 1

	cases := [][]func(*rope.Options){
		{rope.WithTypeNeoX()},
		// {rope.WithTypeNeoX(), rope.WithOriginalContextLength(originalContextLength)},
		// {rope.WithTypeNeoX(), rope.WithOriginalContextLength(originalContextLength), rope.WithExtrapolationFactor(1.)},
		// {rope.WithTypeNeoX(), rope.WithOriginalContextLength(originalContextLength), rope.WithExtrapolationFactor(1.), rope.WithAttentionFactor(0.1*float32(math.Log(float64(ropeScalingFactor))) + 1.)},
		// {rope.WithTypeNeoX(), rope.WithOriginalContextLength(originalContextLength), rope.WithExtrapolationFactor(1.), rope.WithAttentionFactor(float32(math.Sqrt(1. + math.Log(float64(ropeScalingFactor))/float64(originalContextLength))))},
	}

	f32s := make([]float32, tokens*64*headDim)
	for i := range f32s {
		f32s[i] = rand.Float32()
	}

	for _, opts := range cases {
		t.Run("rope", func(t *testing.T) {
			ctx := b.NewContext().Input()
			defer ctx.Close()

			tt := ctx.FromFloatSlice(f32s, headDim, len(f32s)/tokens/headDim, tokens)
			pos := ctx.Arange(0, 100, 1, ml.DTypeI32)

			tt = fast.RoPE(ctx, tt, pos, headDim, float32(ropeTheta), 1/float32(ropeScalingFactor), opts...)
			t.Log("tt:", tt.Shape())
			t.Log(ml.Dump(ctx, tt))
		})
	}
}

func TestActivations(t *testing.T) {
	b := setup(t)

	cases := map[string]func(ml.Context, ml.Tensor) ml.Tensor{
		"tanh": func(ctx ml.Context, x ml.Tensor) ml.Tensor { return x.Tanh(ctx) },
		"gelu": func(ctx ml.Context, x ml.Tensor) ml.Tensor { return x.GELU(ctx) },
		// "quick_gelu": func(ctx ml.Context, x ml.Tensor) ml.Tensor { return x.QuickGELU(ctx) },
		"silu":    func(ctx ml.Context, x ml.Tensor) ml.Tensor { return x.SILU(ctx) },
		"relu":    func(ctx ml.Context, x ml.Tensor) ml.Tensor { return x.RELU(ctx) },
		"sigmoid": func(ctx ml.Context, x ml.Tensor) ml.Tensor { return x.Sigmoid(ctx) },
		"swiglu": func(ctx ml.Context, x ml.Tensor) ml.Tensor {
			alpha := ctx.FromFloatSlice([]float32{1.702}, 1)
			return x.Mul(ctx, alpha).Sigmoid(ctx).Mul(ctx, x)
		},
	}

	shape := []int{2880, 4, 7}
	f32s := make([]float32, mul(shape...))
	for i := range f32s {
		f32s[i] = rand.Float32()
	}

	for name, fn := range cases {
		t.Run(name, func(t *testing.T) {
			ctx := b.NewContext().Input()
			defer ctx.Close()

			tt := ctx.FromFloatSlice(f32s, shape...)
			t.Log("tt:", tt.Shape())

			tt = fn(ctx, tt)
			t.Log("tt:", tt.Shape())
			t.Log(ml.Dump(ctx, tt))
		})
	}
}

func TestChunk(t *testing.T) {
	b := setup(t)

	chunk := func(ctx ml.Context, t ml.Tensor) []ml.Tensor {
		return []ml.Tensor{
			t.View(ctx, 0, t.Dim(0)/2, t.Stride(1), t.Dim(1), t.Stride(2), t.Dim(2)).Contiguous(ctx),
			t.View(ctx, t.Dim(0)/2*t.Stride(0), t.Dim(0)/2, t.Stride(1), t.Dim(1), t.Stride(2), t.Dim(2)).Contiguous(ctx),
		}
	}

	t.Run("chunk", func(t *testing.T) {
		ctx := b.NewContext().Input()
		defer ctx.Close()

		tt := ctx.Arange(0, 4*8, 1, ml.DTypeF32).Reshape(ctx, 8, 4)
		t.Log("tt:", tt.Shape())

		chunks := chunk(ctx, tt)
		t.Log("chunks:", len(chunks))
		t.Log("chunk 0:", ml.Dump(ctx, chunks[0]))
		// t.Log("chunk 1:", ml.Dump(ctx, chunks[1]))
	})
}

func TestSwiglu(t *testing.T) {
	b := setup(t)

	t.Run("swiglu", func(t *testing.T) {
		ctx := b.NewContext().Input()
		defer ctx.Close()

		tt := ctx.Arange(0, 10, 1, ml.DTypeF32)
		t.Log("tt:", tt.Shape())

		tt = tt.Reshape(ctx, 2, 5)
		t.Log("dims", tt.Dim(0), tt.Dim(1), tt.Dim(2))
		t.Log("strides", tt.Stride(0), tt.Stride(1), tt.Stride(2))

		// tt = tt.View(ctx, 0, tt.Dim(0)/2, tt.Stride(1), tt.Dim(1))
		tt = tt.View(ctx, tt.Stride(0), tt.Dim(0)/2, tt.Stride(1), tt.Dim(1))
		t.Log("tt:", tt.Shape())
		t.Log(ml.Dump(ctx, tt.Contiguous(ctx)))
	})
}
