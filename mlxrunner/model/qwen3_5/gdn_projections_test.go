package qwen3_5

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/nn"
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
	if err := bitEqualError(got, want); err != nil {
		t.Fatalf("%s: %v", label, err)
	}
}

func bitEqualError(got, want *mlx.Array) error {
	if got == nil || want == nil {
		return fmt.Errorf("missing numerical comparison: got %v, want %v", got, want)
	}
	if !slices.Equal(got.Dims(), want.Dims()) {
		return fmt.Errorf("shape %v, want %v", got.Dims(), want.Dims())
	}
	got32, want32 := mlx.Contiguous(got, false).AsType(mlx.DTypeFloat32), mlx.Contiguous(want, false).AsType(mlx.DTypeFloat32)
	mlx.Eval(got32, want32)
	gv, wv := got32.Floats(), want32.Floats()
	if len(gv) == 0 || len(gv) != len(wv) {
		return fmt.Errorf("length %d, want nonempty length %d", len(gv), len(wv))
	}
	for i, v := range gv {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.IsNaN(float64(wv[i])) || math.IsInf(float64(wv[i]), 0) {
			return fmt.Errorf("element %d is not finite: got %g, want %g", i, v, wv[i])
		}
		if math.Float32bits(v) != math.Float32bits(wv[i]) {
			return fmt.Errorf("element %d = %g, want %g", i, v, wv[i])
		}
	}
	return nil
}

func TestBitEqualRejectsInvalidComparisons(t *testing.T) {
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
			err := bitEqualError(got, want)
			if (err == nil) != tc.valid {
				t.Fatalf("comparison error %v, want valid=%t", err, tc.valid)
			}
		})
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
