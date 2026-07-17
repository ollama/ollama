package mlx

import "testing"

func TestFastBlockLinearMatchesQuantizedMatmul(t *testing.T) {
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("requires Metal")
		}

		for _, tc := range []struct {
			name   string
			tokens int
			inDim  int
			outDim int
			group  int
			bits   int
			mode   string
		}{
			{name: "nvfp4_shared_down", tokens: 96, inDim: 512, outDim: 2048, group: 16, bits: 4, mode: "nvfp4"},
			{name: "nvfp4_attention_kv", tokens: 96, inDim: 2048, outDim: 1024, group: 16, bits: 4, mode: "nvfp4"},
			{name: "mxfp8_attention_kv", tokens: 96, inDim: 2048, outDim: 1024, group: 32, bits: 8, mode: "mxfp8"},
		} {
			xVals := make([]float32, tc.tokens*tc.inDim)
			wVals := make([]float32, tc.outDim*tc.inDim)
			for i := range xVals {
				xVals[i] = float32((i%31)-15) * 0.01
			}
			for i := range wVals {
				wVals[i] = float32((i%37)-18) * 0.008
			}

			x := FromValues(xVals, 1, tc.tokens, tc.inDim).AsType(DTypeBFloat16)
			w := FromValues(wVals, tc.outDim, tc.inDim).AsType(DTypeBFloat16)
			qw, scales, qbiases := Quantize(w, tc.group, tc.bits, tc.mode)
			Eval(x, qw, scales)
			Pin(x, qw, scales, qbiases)

			got, ok := FastQuantizedLinear(x, qw, scales, nil, tc.group, tc.bits, tc.mode)
			if !ok {
				t.Fatalf("%s: fast linear returned ok=false", tc.name)
			}
			want := QuantizedMatmul(x, qw, scales, qbiases, true, tc.group, tc.bits, tc.mode)
			gotF := got.AsType(DTypeFloat32)
			wantF := want.AsType(DTypeFloat32)
			Eval(gotF, wantF)

			if dims := got.Dims(); len(dims) != 3 || dims[0] != 1 || dims[1] != tc.tokens || dims[2] != tc.outDim {
				t.Fatalf("%s: dims = %v, want [1 %d %d]", tc.name, dims, tc.tokens, tc.outDim)
			}
			assertFloat32Close(t, gotF.Floats(), wantF.Floats(), 2e-2)
			Unpin(x, qw, scales, qbiases)
		}
	})
}

func TestFastQuantizedLinearSmallOutputDimMatchesQuantizedMatmul(t *testing.T) {
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("requires Metal")
		}

		const (
			tokens = 96
			inDim  = 2048
			outDim = 64
		)

		xVals := make([]float32, tokens*inDim)
		wVals := make([]float32, outDim*inDim)
		for i := range xVals {
			xVals[i] = float32((i%31)-15) * 0.01
		}
		for i := range wVals {
			wVals[i] = float32((i%37)-18) * 0.008
		}

		x := FromValues(xVals, 1, tokens, inDim).AsType(DTypeBFloat16)
		w := FromValues(wVals, outDim, inDim).AsType(DTypeBFloat16)
		qw, scales, qbiases := Quantize(w, 16, 4, "nvfp4")
		Eval(x, qw, scales)
		Pin(x, qw, scales, qbiases)
		defer Unpin(x, qw, scales, qbiases)

		got, ok := FastQuantizedLinear(x, qw, scales, nil, 16, 4, "nvfp4")
		if !ok {
			t.Fatal("FastQuantizedLinear returned ok=false for small output dimension")
		}
		want := QuantizedMatmul(x, qw, scales, qbiases, true, 16, 4, "nvfp4")
		gotF := got.AsType(DTypeFloat32)
		wantF := want.AsType(DTypeFloat32)
		Eval(gotF, wantF)

		if dims := got.Dims(); len(dims) != 3 || dims[0] != 1 || dims[1] != tokens || dims[2] != outDim {
			t.Fatalf("dims = %v, want [1 %d %d]", dims, tokens, outDim)
		}
		assertFloat32Close(t, gotF.Floats(), wantF.Floats(), 2e-2)
	})
}

func TestFastQuantizedLinearScaledMatchesSeparateScale(t *testing.T) {
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("requires Metal")
		}

		const (
			tokens = 96
			inDim  = 2048
			outDim = 1024
		)

		xVals := make([]float32, tokens*inDim)
		wVals := make([]float32, outDim*inDim)
		for i := range xVals {
			xVals[i] = float32((i%31)-15) * 0.01
		}
		for i := range wVals {
			wVals[i] = float32((i%37)-18) * 0.008
		}

		x := FromValues(xVals, 1, tokens, inDim).AsType(DTypeBFloat16)
		w := FromValues(wVals, outDim, inDim).AsType(DTypeBFloat16)
		globalScale := FromValue[float32](0.375)
		qw, scales, qbiases := Quantize(w, 16, 4, "nvfp4")
		Eval(x, qw, scales, globalScale)
		Pin(x, qw, scales, qbiases, globalScale)
		defer Unpin(x, qw, scales, qbiases, globalScale)

		got, ok := FastQuantizedLinear(x, qw, scales, globalScale, 16, 4, "nvfp4")
		if !ok {
			t.Fatal("FastQuantizedLinear returned ok=false with global scale")
		}
		base, ok := FastQuantizedLinear(x, qw, scales, nil, 16, 4, "nvfp4")
		if !ok {
			t.Fatal("FastQuantizedLinear returned ok=false")
		}
		want := Mul(base, globalScale).AsType(base.DType())
		gotF := got.AsType(DTypeFloat32)
		wantF := want.AsType(DTypeFloat32)
		Eval(gotF, wantF)

		if dims := got.Dims(); len(dims) != 3 || dims[0] != 1 || dims[1] != tokens || dims[2] != outDim {
			t.Fatalf("dims = %v, want [1 %d %d]", dims, tokens, outDim)
		}
		assertFloat32Close(t, gotF.Floats(), wantF.Floats(), 2e-2)
	})
}
