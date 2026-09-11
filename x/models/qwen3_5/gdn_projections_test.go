package qwen3_5

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

func gdnTestConfig() *Config {
	return &Config{
		LinearNumKeyHeads:   2,
		LinearNumValueHeads: 4,
		LinearKeyHeadDim:    4,
		LinearValueHeadDim:  4,
	}
}

func patternArray(rows, cols int) *mlx.Array {
	values := make([]float32, rows*cols)
	for i := range values {
		values[i] = float32((i*37)%113-56) * 0.03
	}
	return mlx.FromValues(values, rows, cols).AsType(mlx.DTypeBFloat16)
}

func assertBitEqual(t *mlxtest.T, label string, got, want *mlx.Array) {
	t.Helper()
	if got == nil || want == nil {
		if got != want {
			t.Fatalf("%s: got %v, want %v", label, got, want)
		}
		return
	}
	got32, want32 := got.AsType(mlx.DTypeFloat32), want.AsType(mlx.DTypeFloat32)
	mlx.Eval(got32, want32)
	gotValues, wantValues := got32.Floats(), want32.Floats()
	if len(gotValues) != len(wantValues) {
		t.Fatalf("%s: length %d, want %d", label, len(gotValues), len(wantValues))
	}
	for i := range wantValues {
		if math.Float32bits(gotValues[i]) != math.Float32bits(wantValues[i]) {
			t.Fatalf("%s[%d] = %v, want %v", label, i, gotValues[i], wantValues[i])
		}
	}
}

// scatterRows builds the native interleaved tensor from a packed tensor and
// the packed->native row map, inverting what the loader's permutation does.
func scatterRows(packed *mlx.Array, perm []int32) *mlx.Array {
	inverse := make([]int32, len(perm))
	for dst, src := range perm {
		inverse[src] = int32(dst)
	}
	return mlx.Take(packed, mlx.NewArrayInt32(inverse, []int32{int32(len(inverse))}), 0)
}

func TestPackGatedDeltaProjectionsNativeMatchesSplit(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		cfg := gdnTestConfig()
		keyDim := int(cfg.LinearNumKeyHeads * cfg.LinearKeyHeadDim)
		valueDim := int(cfg.LinearNumValueHeads * cfg.LinearValueHeadDim)
		in := 8

		qkvW := patternArray(2*keyDim+valueDim, in)
		zW := patternArray(valueDim, in)
		bW := patternArray(int(cfg.LinearNumValueHeads), in)
		aW := patternArray(int(cfg.LinearNumValueHeads), in)

		fromSplitQKVZ, fromSplitBA, err := packGatedDeltaProjections(
			nn.NewLinear(qkvW, nil), nn.NewLinear(zW, nil),
			nn.NewLinear(bW, nil), nn.NewLinear(aW, nil),
			nil, nil, cfg,
		)
		if err != nil {
			t.Fatal(err)
		}

		packedQKVZ := mlx.Concatenate([]*mlx.Array{qkvW, zW}, 0)
		packedBA := mlx.Concatenate([]*mlx.Array{bW, aW}, 0)
		nativeQKVZ := scatterRows(packedQKVZ, nativeQKVZPerm(cfg))
		nativeBA := scatterRows(packedBA, nativeBAPerm(cfg))

		fromNativeQKVZ, fromNativeBA, err := packGatedDeltaProjections(
			nil, nil, nil, nil,
			nn.NewLinear(nativeQKVZ, nil), nn.NewLinear(nativeBA, nil),
			cfg,
		)
		if err != nil {
			t.Fatal(err)
		}

		assertBitEqual(t, "qkvz", fromNativeQKVZ.(*nn.Linear).Weight, fromSplitQKVZ.(*nn.Linear).Weight)
		assertBitEqual(t, "ba", fromNativeBA.(*nn.Linear).Weight, fromSplitBA.(*nn.Linear).Weight)
	})
}

func TestConcatProjectionPairQuantized(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		in := 64
		hi := nn.NewQuantizedLinear(patternArray(16, in).AsType(mlx.DTypeFloat32), nil, 32, 4, "affine")
		lo := nn.NewQuantizedLinear(patternArray(8, in).AsType(mlx.DTypeFloat32), nil, 32, 4, "affine")

		packed, err := concatProjectionPair(hi, lo)
		if err != nil {
			t.Fatal(err)
		}
		q, ok := packed.(*nn.QuantizedLinear)
		if !ok {
			t.Fatalf("packed projection is %T, want *nn.QuantizedLinear", packed)
		}
		assertBitEqual(t, "weight", q.Weight, mlx.Concatenate([]*mlx.Array{hi.Weight, lo.Weight}, 0))
		assertBitEqual(t, "scales", q.Scales, mlx.Concatenate([]*mlx.Array{hi.Scales, lo.Scales}, 0))
	})
}

func TestConcatProjectionPairMixedFallsBackToDense(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		in := 64
		hi := nn.NewQuantizedLinear(patternArray(16, in).AsType(mlx.DTypeFloat32), nil, 32, 4, "affine")
		loW := patternArray(8, in)

		packed, err := concatProjectionPair(hi, nn.NewLinear(loW, nil))
		if err != nil {
			t.Fatal(err)
		}
		dense, ok := packed.(*nn.Linear)
		if !ok {
			t.Fatalf("packed projection is %T, want *nn.Linear", packed)
		}
		want := mlx.Concatenate([]*mlx.Array{
			mlx.Dequantize(hi.Weight, hi.Scales, hi.QBiases, hi.GroupSize, hi.Bits, hi.Mode, nil),
			loW.AsType(mlx.DTypeFloat16),
		}, 0)
		if dense.Weight.Dim(0) != 24 {
			t.Fatalf("packed rows = %d, want 24", dense.Weight.Dim(0))
		}
		assertBitEqual(t, "weight", dense.Weight, want.AsType(dense.Weight.DType()))
	})
}
