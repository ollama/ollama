package ggml

import (
	"fmt"
	"log/slog"
	"math"
	"math/rand"
	"os"
	"strings"
	"testing"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"

	fsggml "github.com/ollama/ollama/fs/ggml"
)

/*
	To get GPUs loading in these tests on windows...

	$env:OLLAMA_LIBRARY_PATH="$(pwd)\build\lib\ollama"
	$env:PATH="$(pwd)\build\lib\ollama;$env:PATH"

	go test .\ml\backend\ggml\... -run TestMXFP4 -model llama3.2

*/

var (
	/*
		MXFP4 blocks - 17 bytes long
			- 8-bit Scale: E8M0
			- (32) 4-bit Elements: E2M1

		if E > 0, then (-1)^S * 2^(E-bias) * (1 + 2^-1 * M). This is a normal number
		if E = 0, then (-1)^S * 2^(1-bias) * (0 + 2^-1 * M). This is a subnormal number
		S: sign
		E: exponent
		M: mantissa
		bias: 1
		                  EE M
		Max normal      S 11 1 = ± 2^2 × 1.5 = ± 6.0
		Min normal      S 01 0 = ± 2^0 × 1.0 = ± 1.0
		Max subnorm     S 00 1 = ± 2^0 × 0.5 = ± 0.5
		Min subnorm     S 00 1 = ± 2^0 × 0.5 = ± 0.5

		Scale E8M0
		bias: 127
		NaN: 0xff

	*/

	// E2M1 values
	mxfp4_vals = []float32{
		0.0,  // 0 00 0 = 0x0
		0.5,  // 0 00 1 = 0x1
		1.0,  // 0 01 0 = 0x2
		1.5,  // 0 01 1 = 0x3
		2.0,  // 0 10 0 = 0x4
		3.0,  // 0 10 1 = 0x5
		4.0,  // 0 11 0 = 0x6
		6.0,  // 0 11 1 = 0x7
		0.0,  // 1 00 0 = 0x8
		-0.5, // 1 00 1 = 0x9
		-1.0, // 1 01 0 = 0xa
		-1.5, // 1 01 1 = 0xb
		-2.0, // 1 10 0 = 0xc
		-3.0, // 1 10 1 = 0xd
		-4.0, // 1 11 0 = 0xe
		-6.0, // 1 11 1 = 0xf
	}
)

func init() {
	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))
}

func TestMXFP4Ops(t *testing.T) {
	b := setup(t)

	t.Run("mulmatid", func(t *testing.T) {

		// Use exact values that are supported without scaling so we can compare against an fp32 tensor
		t.Run("exact", func(t *testing.T) {
			r := rand.New(rand.NewSource(0))
			ctx := b.NewContext().Input()
			defer ctx.Close()

			data := [64 * 4 * 8]float32{}
			for i := range data {
				data[i] = mxfp4_vals[r.Int()%len(mxfp4_vals)]
			}
			mxData := Quantize(fsggml.TensorTypeMXFP4, data[:], []uint64{uint64(len(data))})
			dtype := ml.DTypeMXFP4
			t1 := ctx.(*Context).FromBytes(dtype, mxData, 64, 4, 8)
			t1f := ctx.(*Context).FromFloatSlice(data[:], 64, 4, 8)

			// arange equiv
			d2 := [64 * 1 * 8]float32{}
			for i := range d2 {
				d2[i] = float32(i)
			}
			t2 := ctx.(*Context).FromFloatSlice(d2[:], 64, 1, 8)

			// arange equiv
			d3 := [1 * 8]int32{}
			for i := range d3 {
				d3[i] = int32(i)
			}
			t3 := ctx.(*Context).FromIntSlice(d3[:], 1, 8)

			// t.Log("calling MulmatID")
			t4 := t1.MulmatID(ctx, t2, t3)
			t4f := t1f.MulmatID(ctx, t2, t3)
			d4 := ml.Dump(ctx, t4)
			d4f := ml.Dump(ctx, t4f)
			if d4 != d4f {
				t.Fatalf("expected (f32): \n%s\n\n but got (mxfp4): \n%s", d4f, d4)
			}
			// t.Logf("MulmatID results matched:\n%s", d4)
		})

		t.Run("range", func(t *testing.T) {
			r := rand.New(rand.NewSource(0))
			ctx := b.NewContext().Input()
			defer ctx.Close()
			const s0 = 32
			const s1 = 2
			const s2 = 4
			data := [s0 * s1 * s2]float32{}
			inTotal := float32(0)
			for i := range data {
				data[i] = float32(i)
				inTotal += float32(i)
			}
			mxData := Quantize(fsggml.TensorTypeMXFP4, data[:], []uint64{uint64(len(data))})
			// Reconvert back to floats to remove the quantization fidelity loss for comparison
			dataf := ConvertToF32(mxData, uint32(fsggml.TensorTypeMXFP4), uint64(len(data)))
			dtype := ml.DTypeMXFP4
			t1 := ctx.(*Context).FromBytes(dtype, mxData, s0, s1, s2)
			t1f := ctx.(*Context).FromFloatSlice(dataf, s0, s1, s2)
			for i := range len(data) / 32 {
				vals := [32]string{}
				for j := range vals {
					vals[j] = fmt.Sprintf("%0.2f", dataf[i*32+j])
				}
				t.Logf("  t1[%s]\n", strings.Join(vals[:], ", "))
			}

			d2 := [s0]float32{}
			for i := range d2 {
				// d2[i] = float32(i)
				d2[i] = float32(r.Float32())
			}
			for i := range len(d2) / s0 {
				vals := [s0]string{}
				for j := range vals {
					vals[j] = fmt.Sprintf("%0.2f", d2[i*s0+j])
				}
				t.Logf("  t2[%s]\n", strings.Join(vals[:], ", "))
			}
			t2 := ctx.(*Context).FromFloatSlice(d2[:], s0)

			// arange equiv
			d3 := [4]int32{}
			for i := range d3 {
				d3[i] = int32(i)
			}
			t3 := ctx.(*Context).FromIntSlice(d3[:], 4)

			// t.Log("calling Mulmat")
			// t3 := t1.Mulmat(ctx, t2)
			// t3f := t1f.Mulmat(ctx, t2)
			t4 := t1.MulmatID(ctx, t2, t3)
			t4f := t1f.MulmatID(ctx, t2, t3)
			d4 := ml.Dump(ctx, t4, ml.DumpWithPrecision(3))
			d4f := ml.Dump(ctx, t4f, ml.DumpWithPrecision(3))
			r4 := t4.Floats()
			r4f := t4f.Floats()
			sim := cosineSimilarity(r4, r4f)
			if sim < 0.99 {
				t.Logf("expected (f32): \n%s\n\n but got (mxfp4): \n%s", d4f, d4)
				t.Fatalf("failed similarity test: %f", sim)
			}
			t.Logf("similarity: %f", sim)

			if d4 != d4f {
				t.Fatalf("expected (f32): \n%s\n\n but got (mxfp4): \n%s", d4f, d4)
			}
		})

		// Use data file(s) with real data
		t.Run("example", func(t *testing.T) {
			ctx := b.NewContext().Input()
			defer ctx.Close()

			data0, err := os.ReadFile("mlp-gateup.bin")
			if err != nil {
				t.Skip("missing mlp-gateup.bin file, skipping test")
			}
			data1, err := os.ReadFile("hidden-states.bin")
			if err != nil {
				t.Skip("missing hidden-states.bin file, skipping test")
			}
			data2, err := os.ReadFile("selected-experts.bin")
			if err != nil {
				t.Skip("missing selected-experts.bin file, skipping test")
			}

			dtype := ml.DTypeMXFP4
			data0f := ConvertToF32(data0, 39, 2880*5760*32)
			t1 := ctx.(*Context).FromBytes(dtype, data0, 2880, 5760, 32)
			t1f := ctx.(*Context).FromFloatSlice(data0f, 2880, 5760, 32)

			// t.Logf("f32: \n%s", ml.Dump(ctx, t1f))

			t2 := ctx.(*Context).FromBytes(ml.DTypeF32, data1, 2880, 1, 7)
			// t.Logf("f32: \n%s", ml.Dump(ctx, t2))

			t3 := ctx.(*Context).FromBytes(ml.DTypeI32, data2, 4, 7)

			// t.Log("calling MulmatID")
			t4 := t1.MulmatID(ctx, t2, t3)
			t4f := t1f.MulmatID(ctx, t2, t3)

			d4 := ml.Dump(ctx, t4)
			d4f := ml.Dump(ctx, t4f)

			r4 := t4.Floats()
			r4f := t4f.Floats()
			sim := cosineSimilarity(r4, r4f)
			if sim < 0.99 {
				t.Fatalf("failed similarity test: %f", sim)
			}
			t.Logf("similarity: %f", sim)

			if d4 != d4f {
				t.Fatalf("expected (f32): \n%s\n\n but got (mxfp4): \n%s", d4f, d4)
			}
			// t.Logf("MulmatID results matched:\n%s", d4)
		})
	})

	t.Run("mm", func(t *testing.T) {

		t.Run("example", func(t *testing.T) {
			r := rand.New(rand.NewSource(0))
			ctx := b.NewContext().Input()
			defer ctx.Close()

			data0, err := os.ReadFile("mlp-gateup.bin")
			if err != nil {
				t.Skip("missing mlp-gateup.bin file, skipping test")
			}
			data1 := [2880 * 1 * 32]float32{}
			for i := range data1 {
				data1[i] = float32(r.Float32())
			}

			dtype := ml.DTypeMXFP4
			data0f := ConvertToF32(data0, 39, 2880*5760*32)
			t1 := ctx.(*Context).FromBytes(dtype, data0, 2880, 5760, 32)
			t1f := ctx.(*Context).FromFloatSlice(data0f, 2880, 5760, 32)

			// t.Logf("f32: \n%s", ml.Dump(ctx, t1f))

			t2 := ctx.(*Context).FromFloatSlice(data1[:], 2880, 1, 32)

			t4 := t1.Mulmat(ctx, t2)
			t4f := t1f.Mulmat(ctx, t2)

			d4 := ml.Dump(ctx, t4)
			d4f := ml.Dump(ctx, t4f)

			r4 := t4.Floats()
			r4f := t4f.Floats()
			sim := cosineSimilarity(r4, r4f)
			if sim < 0.99 {
				t.Fatalf("failed similarity test: %f", sim)
			}
			t.Logf("similarity: %f", sim)

			if d4 != d4f {
				t.Fatalf("expected (f32): \n%s\n\n but got (mxfp4): \n%s", d4f, d4)
			}
			// t.Logf("Mulmat results matched:\n%s", d4)
		})

		t.Run("exact/2d", func(t *testing.T) {
			r := rand.New(rand.NewSource(0))
			ctx := b.NewContext().Input()
			defer ctx.Close()

			data := [32 * 4]float32{}
			for i := range data {
				data[i] = mxfp4_vals[r.Int()%len(mxfp4_vals)]
			}
			// for i := range 4 {
			// 	vals := [32]string{}
			// 	for j := range vals {
			// 		vals[j] = fmt.Sprintf("%0.2f", data[i*32+j])
			// 	}
			// 	t.Logf("  [%s]\n", strings.Join(vals[:], ", "))
			// }
			mxData := Quantize(fsggml.TensorTypeMXFP4, data[:], []uint64{uint64(len(data))})
			// for i := range len(mxData) / 17 {
			// 	vals := [17]string{}
			// 	for j := range vals {
			// 		vals[j] = fmt.Sprintf("%0.2x", mxData[i*17+j])
			// 	}
			// 	t.Logf("  %s\n", strings.Join(vals[:], ", "))
			// }
			dtype := ml.DTypeMXFP4
			t1 := ctx.(*Context).FromBytes(dtype, mxData, 32, 4)
			t1f := ctx.(*Context).FromFloatSlice(data[:], 32, 4)

			d2 := [32 * 4]float32{}
			for i := range d2 {
				d2[i] = 2.0
			}
			t2 := ctx.(*Context).FromFloatSlice(d2[:], 32, 4)

			t3f := t1f.Mulmat(ctx, t2)
			t3 := t1.Mulmat(ctx, t2)
			d3 := ml.Dump(ctx, t3)
			d3f := ml.Dump(ctx, t3f)
			if d3 != d3f {
				t.Fatalf("expected (f32): \n%s\n\n but got (mxfp4): \n%s", d3f, d3)
			}
		})

		t.Run("range/2d", func(t *testing.T) {
			r := rand.New(rand.NewSource(0))
			ctx := b.NewContext().Input()
			defer ctx.Close()
			const s0 = 32
			const s1 = 4
			data := [s0 * s1]float32{}
			inTotal := float32(0)
			for i := range data {
				data[i] = float32(i)
				inTotal += float32(i)
			}
			mxData := Quantize(fsggml.TensorTypeMXFP4, data[:], []uint64{uint64(len(data))})
			// Reconvert back to floats to remove the quantization fidelity loss for comparison
			dataf := ConvertToF32(mxData, uint32(fsggml.TensorTypeMXFP4), uint64(len(data)))
			dtype := ml.DTypeMXFP4
			t1 := ctx.(*Context).FromBytes(dtype, mxData, s0, s1)
			t1f := ctx.(*Context).FromFloatSlice(dataf, s0, s1)
			for i := range len(data) / 32 {
				vals := [32]string{}
				for j := range vals {
					vals[j] = fmt.Sprintf("%0.2f", dataf[i*32+j])
				}
				t.Logf("  t1[%s]\n", strings.Join(vals[:], ", "))
			}

			d2 := [s0 * 4]float32{}
			for i := range d2 {
				// d2[i] = float32(i)
				d2[i] = float32(r.Float32())
			}
			for i := range len(d2) / s0 {
				vals := [s0]string{}
				for j := range vals {
					vals[j] = fmt.Sprintf("%0.2f", d2[i*s0+j])
				}
				t.Logf("  t2[%s]\n", strings.Join(vals[:], ", "))
			}

			t2 := ctx.(*Context).FromFloatSlice(d2[:], s0, 4)

			// t.Log("calling Mulmat")
			t3 := t1.Mulmat(ctx, t2)
			t3f := t1f.Mulmat(ctx, t2)
			d3 := ml.Dump(ctx, t3, ml.DumpWithPrecision(3))
			d3f := ml.Dump(ctx, t3f, ml.DumpWithPrecision(3))
			r3 := t3.Floats()
			r3f := t3f.Floats()
			sim := cosineSimilarity(r3, r3f)
			if sim < 0.99 {
				t.Logf("expected (f32): \n%s\n\n but got (mxfp4): \n%s", d3f, d3)
				t.Fatalf("failed similarity test: %f", sim)
			}
			t.Logf("similarity: %f", sim)
			if d3 != d3f {
				t.Fatalf("expected (f32): \n%s\n\n but got (mxfp4): \n%s", d3f, d3)
			}
		})

		t.Run("range/3d", func(t *testing.T) {
			ctx := b.NewContext().Input()
			defer ctx.Close()
			data := [32 * 4 * 2]float32{}
			inTotal := float32(0)
			for i := range data {
				data[i] = float32(i)
				inTotal += float32(i)
			}
			mxData := Quantize(fsggml.TensorTypeMXFP4, data[:], []uint64{uint64(len(data))})
			dtype := ml.DTypeMXFP4
			// Reconvert back to floats to remove the quantization fidelity loss for comparison
			dataf := ConvertToF32(mxData, uint32(fsggml.TensorTypeMXFP4), uint64(len(data)))
			t1 := ctx.(*Context).FromBytes(dtype, mxData, 32, 4, 2)
			t1f := ctx.(*Context).FromFloatSlice(dataf, 32, 4, 2)

			d2 := [32 * 4 * 2]float32{}
			for i := range d2 {
				d2[i] = 2.0
			}
			t2 := ctx.(*Context).FromFloatSlice(d2[:], 32, 4, 2)

			t.Log("calling Mulmat")
			t3 := t1.Mulmat(ctx, t2)
			t3f := t1f.Mulmat(ctx, t2)
			d3 := ml.Dump(ctx, t3)
			d3f := ml.Dump(ctx, t3f)
			r3 := t3.Floats()
			r3f := t3f.Floats()
			sim := cosineSimilarity(r3, r3f)
			if sim < 0.99 {
				t.Logf("expected (f32): \n%s\n\n but got (mxfp4): \n%s", d3f, d3)
				t.Fatalf("failed similarity test: %f", sim)
			}
			t.Logf("similarity: %f", sim)
			if d3 != d3f {
				t.Fatalf("expected (f32): \n%s\n\n but got (mxfp4): \n%s", d3f, d3)
			}
		})
	})
}
func TestMXFP4Simple(t *testing.T) {
	b := setup(t)

	t.Run("fixed", func(t *testing.T) {
		ctx := b.NewContext().Input()
		defer ctx.Close()

		data := [32 * 2]float32{
			2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
			2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
		}
		mxData := Quantize(fsggml.TensorTypeMXFP4, data[:], []uint64{uint64(len(data))})
		dtype := ml.DTypeMXFP4
		// Reconvert back to floats to remove the quantization fidelity loss for comparison
		dataf := ConvertToF32(mxData, uint32(fsggml.TensorTypeMXFP4), uint64(len(data)))
		t1 := ctx.(*Context).FromBytes(dtype, mxData, 32, 2)
		t1f := ctx.(*Context).FromFloatSlice(dataf, 32, 2)

		d2 := [32 * 2]float32{
			// 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		}
		t2 := ctx.(*Context).FromFloatSlice(d2[:], 32, 2)

		t.Log("calling Mulmat")
		t3f := t1f.Mulmat(ctx, t2)
		t3 := t1.Mulmat(ctx, t2)
		d3 := ml.Dump(ctx, t3)
		d3f := ml.Dump(ctx, t3f)
		if d3 != d3f {
			t.Fatalf("expected (f32): \n%s\n\n but got (mxfp4): \n%s", d3f, d3)
		}
		t.Logf("result (mxfp4): \n%s", d3)
	})

}

func TestMXFP4Conversion(t *testing.T) {
	t.Run("quantize/exact", func(t *testing.T) {
		r := rand.New(rand.NewSource(0))

		data := [32 * 4]float32{}
		for i := range data {
			data[i] = mxfp4_vals[r.Int()%len(mxfp4_vals)] * 0.1
		}
		mxData := Quantize(fsggml.TensorTypeMXFP4, data[:], []uint64{uint64(len(data))})
		newData := ConvertToF32(mxData, uint32(fsggml.TensorTypeMXFP4), uint64(len(data)))

		if len(data) != len(newData) {
			t.Fatalf("length mismatch.  started with %d but got %d", len(data), len(newData))
		}
		for i := range data {
			if data[i] != newData[i] {
				t.Logf("started with: %v", data)
				t.Logf("got         : %v", newData)
				t.Fatalf("mismatched data starting at offset %d started with %f but got %f", i, data[i], newData[i])
			}
		}
	})
	t.Run("quantize/arange", func(t *testing.T) {
		data := [32 * 8]float32{}
		for i := range data {
			data[i] = float32(i) // / float32(6.0)
		}
		mxData := Quantize(fsggml.TensorTypeMXFP4, data[:], []uint64{uint64(len(data))})
		newData := ConvertToF32(mxData, uint32(fsggml.TensorTypeMXFP4), uint64(len(data)))

		if len(data) != len(newData) {
			t.Fatalf("length mismatch.  started with %d but got %d", len(data), len(newData))
		}
		sim := cosineSimilarity(data[:], newData)
		if sim < 0.99 {
			t.Fatalf("failed similarity test: %f", sim)
		}
		t.Logf("similarity: %f", sim)
	})
}

func TestMulmat(t *testing.T) {
	b := setup(t)

	t.Run("fixed", func(t *testing.T) {
		/* Equivalent to
		import numpy as np
		m1 = np.array([[1, 2], [3, 4]])
		m2 = np.array([[5, 6], [7, 8]])
		print(np.matmul(m1, m2))
		*/

		ctx := b.NewContext().Input()
		defer ctx.Close()

		// d1 := [2 * 2]float32{5, 6, 7, 8}
		d1 := [2 * 2]float32{1, 0, 0, 0}
		t1 := ctx.(*Context).FromFloatSlice(d1[:], 2, 2)

		// d2 := [2 * 2]float32{1, 2, 3, 4}
		d2 := [2 * 2]float32{2, 0, 0, 0}
		t2 := ctx.(*Context).FromFloatSlice(d2[:], 2, 2)

		t.Log("calling Mulmat")
		t3 := t1.Mulmat(ctx, t2)
		t.Logf("Results: \n%s", ml.Dump(ctx, t3))
	})
}

func dotProduct[V float32 | float64](v1, v2 []V) V {
	var result V = 0
	for i := range v1 {
		result += v1[i] * v2[i]
	}
	return result
}

func magnitude[V float32 | float64](v []V) V {
	var result V = 0
	for _, val := range v {
		result += val * val
	}
	return V(math.Sqrt(float64(result)))
}

func cosineSimilarity[V float32 | float64](v1, v2 []V) V {
	return dotProduct(v1, v2) / (magnitude(v1) * magnitude(v2))
}
