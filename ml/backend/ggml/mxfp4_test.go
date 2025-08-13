package ggml

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/ollama/ollama/ml"

	fsggml "github.com/ollama/ollama/fs/ggml"
)

/*
	To get GPUs loading in these tests on windows...

	$env:OLLAMA_LIBRARY_PATH="$(pwd)\build\lib\ollama"
	$env:PATH="$(pwd)\build\lib\ollama;$env:PATH"

	go test .\ml\backend\ggml\... -run TestMXFP4
*/

// MXFP4 reference: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

// E2M1 values
var mxfp4_vals = []float32{
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

func TestMXFP4Ops(t *testing.T) {
	b := setup(t)
	for _, useGPU := range []bool{false, true} {
		useGPU := useGPU
		var label string
		if useGPU {
			label = "gpu"
		} else {
			label = "cpu"
		}
		t.Run(label, func(t *testing.T) {
			t.Run("mulmatid", func(t *testing.T) {
				// Use exact values that are supported without scaling so we can compare against an fp32 tensor
				t.Run("exact", func(t *testing.T) {
					r := rand.New(rand.NewSource(0))
					ctx := initContextOrSkip(t, b, useGPU)
					const s00 = 64
					const s01 = 1
					const s02 = 2
					const s10 = s00
					const s11 = 1
					const s12 = 1
					// const s00 = 2880
					// const s01 = 5760
					// const s02 = 32
					// const s10 = s00
					// const s11 = 1
					// const s12 = 64

					data := [s00 * s01 * s02]float32{}
					for i := range data {
						data[i] = mxfp4_vals[r.Int()%len(mxfp4_vals)]
					}
					mxData := Quantize(fsggml.TensorTypeMXFP4, data[:], []uint64{uint64(len(data))})
					dtype := ml.DTypeMXFP4
					t1 := ctx.(*Context).FromBytes(dtype, mxData, s00, s01, s02)
					t1f := ctx.(*Context).FromFloatSlice(data[:], s00, s01, s02)
					// for i := range len(data) / 32 { // MXFP4 block size
					// 	vals := [32]string{}
					// 	for j := range vals {
					// 		vals[j] = fmt.Sprintf("%0.2f", data[i*32+j])
					// 	}
					// 	t.Logf("  t1[%s]\n", strings.Join(vals[:], ", "))
					// }

					// random 0-1 float
					d2 := [s10 * s11 * s12]float32{}
					for i := range d2 {
						d2[i] = float32(r.Float32())
					}
					// for i := range len(d2) / s10 {
					// 	vals := [s10]string{}
					// 	for j := range vals {
					// 		vals[j] = fmt.Sprintf("%0.2f", d2[i*s10+j])
					// 	}
					// 	t.Logf("  t2[%s]\n", strings.Join(vals[:], ", "))
					// }
					t2 := ctx.(*Context).FromFloatSlice(d2[:], s10, s11, s12)

					d3 := [4 * s12]int32{}
					for i := range d3 {
						d3[i] = int32(i) % s02
					}
					t3 := ctx.(*Context).FromIntSlice(d3[:], 4, s12)

					// t.Log("calling MulmatID")
					t4 := t1.MulmatID(ctx, t2, t3)
					t4f := t1f.MulmatID(ctx, t2, t3)
					d4 := ml.Dump(ctx, t4, ml.DumpWithPrecision(2)) // lower precision for CPU accuracy
					d4f := ml.Dump(ctx, t4f, ml.DumpWithPrecision(2))
					if d4 != d4f {
						t.Fatalf("expected (f32): \n%s\n\n but got (mxfp4): \n%s", d4f, d4)
					}
					// t.Logf("MulmatID results matched:\n%s", d4)
				})

				t.Run("range", func(t *testing.T) {
					r := rand.New(rand.NewSource(0))
					ctx := initContextOrSkip(t, b, useGPU)
					const s0 = 64
					const s1 = 2
					const s2 = 4
					const idlen = 4
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
					// for i := range len(data) / 32 {
					// 	vals := [32]string{}
					// 	for j := range vals {
					// 		vals[j] = fmt.Sprintf("%0.2f", dataf[i*32+j])
					// 	}
					// 	t.Logf("  t1[%s]\n", strings.Join(vals[:], ", "))
					// }

					d2 := [s0]float32{}
					for i := range d2 {
						// d2[i] = float32(i)
						d2[i] = float32(r.Float32())
					}
					// for i := range len(d2) / s0 {
					// 	vals := [s0]string{}
					// 	for j := range vals {
					// 		vals[j] = fmt.Sprintf("%0.2f", d2[i*s0+j])
					// 	}
					// 	t.Logf("  t2[%s]\n", strings.Join(vals[:], ", "))
					// }
					t2 := ctx.(*Context).FromFloatSlice(d2[:], s0)

					// TODO - there might be a CUDA bug here...
					d3 := [idlen]int32{1, 1, 2, 3}
					// for i := range d3 {
					// 	d3[i] = int32(i) % s2
					// 	t.Logf("%d] %d", i, d3[i])
					// }
					t3 := ctx.(*Context).FromIntSlice(d3[:], idlen)

					// t.Log("calling Mulmat")
					t4 := t1.MulmatID(ctx, t2, t3)
					t4f := t1f.MulmatID(ctx, t2, t3)
					// Metal has some drift so use reduced precision for dump comparisons
					d4 := ml.Dump(ctx, t4, ml.DumpWithPrecision(2))
					d4f := ml.Dump(ctx, t4f, ml.DumpWithPrecision(2))
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
					// t.Logf("mxfp4 result\n%s", d4)
				})
				t.Run("random", func(t *testing.T) {
					r := rand.New(rand.NewSource(0))
					ctx := initContextOrSkip(t, b, useGPU)
					const s00 = 2880
					const s01 = 5760
					const s02 = 32
					const s10 = s00
					const s11 = 1
					const s12 = 64
					const idlen = 4

					data := [s00 * s01 * s02]float32{}
					for i := range data {
						data[i] = float32(r.Float32() * 10.0)
					}
					mxData := Quantize(fsggml.TensorTypeMXFP4, data[:], []uint64{uint64(len(data))})
					// Reconvert back to floats to remove the quantization fidelity loss for comparison
					dataf := ConvertToF32(mxData, uint32(fsggml.TensorTypeMXFP4), uint64(len(data)))
					dtype := ml.DTypeMXFP4
					t1 := ctx.(*Context).FromBytes(dtype, mxData, s00, s01, s02)
					t1f := ctx.(*Context).FromFloatSlice(dataf, s00, s01, s02)
					// for i := range len(data) / 32 {
					// 	vals := [32]string{}
					// 	for j := range vals {
					// 		vals[j] = fmt.Sprintf("%0.2f", dataf[i*32+j])
					// 	}
					// 	t.Logf("  t1[%s]\n", strings.Join(vals[:], ", "))
					// }

					d2 := [s10 * s11 * s12]float32{}
					for i := range d2 {
						// d2[i] = float32(i)
						d2[i] = float32(r.Float32())
					}
					// for i := range len(d2) / s0 {
					// 	vals := [s0]string{}
					// 	for j := range vals {
					// 		vals[j] = fmt.Sprintf("%0.2f", d2[i*s0+j])
					// 	}
					// 	t.Logf("  t2[%s]\n", strings.Join(vals[:], ", "))
					// }
					t2 := ctx.(*Context).FromFloatSlice(d2[:], s10, s11, s12)

					// arange equiv
					d3 := [idlen * s12]int32{}
					for i := range d3 {
						d3[i] = int32(i) % s02
					}
					t3 := ctx.(*Context).FromIntSlice(d3[:], idlen, s12)

					// t.Log("calling Mulmat")
					// t3 := t1.Mulmat(ctx, t2)
					// t3f := t1f.Mulmat(ctx, t2)
					t4 := t1.MulmatID(ctx, t2, t3)
					t4f := t1f.MulmatID(ctx, t2, t3)
					// Metal and CPU have some drift so use reduced precision for dump comparisons
					d4 := ml.Dump(ctx, t4, ml.DumpWithPrecision(1))
					d4f := ml.Dump(ctx, t4f, ml.DumpWithPrecision(1))
					// t.Logf("mxfp4 data: \n%s", d4)
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
				t.Run("example_7", func(t *testing.T) {
					ctx := initContextOrSkip(t, b, useGPU)
					data0, err := os.ReadFile("mlp-gateup.bin")
					if err != nil {
						t.Skip("missing mlp-gateup.bin file, skipping test")
					}
					data1, err := os.ReadFile("hidden-states-7.bin")
					if err != nil {
						t.Skip("missing hidden-states.bin file, skipping test")
					}
					data2, err := os.ReadFile("selected-experts-7.bin")
					if err != nil {
						t.Skip("missing selected-experts.bin file, skipping test")
					}

					dtype := ml.DTypeMXFP4
					data0f := ConvertToF32(data0, uint32(fsggml.TensorTypeMXFP4), 2880*5760*32)
					t1 := ctx.(*Context).FromBytes(dtype, data0, 2880, 5760, 32)
					t1f := ctx.(*Context).FromFloatSlice(data0f, 2880, 5760, 32)

					// t.Logf("f32: \n%s", ml.Dump(ctx, t1f))

					t2 := ctx.(*Context).FromBytes(ml.DTypeF32, data1, 2880, 1, 7)
					// t.Logf("hidden-state: \n%s", ml.Dump(ctx, t2))

					t3 := ctx.(*Context).FromBytes(ml.DTypeI32, data2, 4, 7)
					// t.Logf("experts: \n%s", ml.Dump(ctx, t3))

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

				// Use data file(s) with real data
				t.Run("example_384", func(t *testing.T) {
					ctx := initContextOrSkip(t, b, useGPU)
					data0, err := os.ReadFile("mlp-gateup.bin")
					if err != nil {
						t.Skip("missing mlp-gateup.bin file, skipping test")
					}
					data1, err := os.ReadFile("hidden-states-384.bin")
					if err != nil {
						t.Skip("missing hidden-states.bin file, skipping test")
					}
					data2, err := os.ReadFile("selected-experts-384.bin")
					if err != nil {
						t.Skip("missing selected-experts.bin file, skipping test")
					}

					dtype := ml.DTypeMXFP4
					data0f := ConvertToF32(data0, uint32(fsggml.TensorTypeMXFP4), 2880*5760*32)
					t1 := ctx.(*Context).FromBytes(dtype, data0, 2880, 5760, 32)
					t1f := ctx.(*Context).FromFloatSlice(data0f, 2880, 5760, 32)

					// t.Logf("f32: \n%s", ml.Dump(ctx, t1f))

					t2 := ctx.(*Context).FromBytes(ml.DTypeF32, data1, 2880, 1, 384)
					// t.Logf("hidden-state: \n%s", ml.Dump(ctx, t2))

					t3 := ctx.(*Context).FromBytes(ml.DTypeI32, data2, 4, 384)
					// t.Logf("experts: \n%s", ml.Dump(ctx, t3))

					// t.Log("calling MulmatID")
					t4 := t1.MulmatID(ctx, t2, t3)
					t4f := t1f.MulmatID(ctx, t2, t3)

					d4 := ml.Dump(ctx, t4, ml.DumpWithPrecision(3))
					d4f := ml.Dump(ctx, t4f, ml.DumpWithPrecision(3))

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

				// Use data file(s) with real data
				t.Run("example_1d", func(t *testing.T) {
					r := rand.New(rand.NewSource(0))
					ctx := initContextOrSkip(t, b, useGPU)
					data0, err := os.ReadFile("mlp-gateup.bin")
					if err != nil {
						t.Skip("missing mlp-gateup.bin file, skipping test")
					}

					dtype := ml.DTypeMXFP4
					data0f := ConvertToF32(data0, uint32(fsggml.TensorTypeMXFP4), 2880*5760*32)
					t1 := ctx.(*Context).FromBytes(dtype, data0, 2880, 5760, 32)
					t1f := ctx.(*Context).FromFloatSlice(data0f, 2880, 5760, 32)

					// t.Logf("f32: \n%s", ml.Dump(ctx, t1f))
					data1 := [2880]float32{}
					for i := range data1 {
						data1[i] = float32(r.Float32())
					}

					t2 := ctx.(*Context).FromFloatSlice(data1[:], 2880)
					// t.Logf("hidden-state: \n%s", ml.Dump(ctx, t2))
					data2 := [4]int32{
						12, 30, 17, 7,
						// 7, 17, 12, 30,
					}

					t3 := ctx.(*Context).FromIntSlice(data2[:], 4)
					// t.Logf("experts: \n%s", ml.Dump(ctx, t3))

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
					ctx := initContextOrSkip(t, b, useGPU)
					data0, err := os.ReadFile("mlp-gateup.bin")
					if err != nil {
						t.Skip("missing mlp-gateup.bin file, skipping test")
					}
					data1 := [2880 * 1 * 32]float32{}
					for i := range data1 {
						data1[i] = float32(r.Float32())
					}

					dtype := ml.DTypeMXFP4
					data0f := ConvertToF32(data0, uint32(fsggml.TensorTypeMXFP4), 2880*5760*32)
					t1 := ctx.(*Context).FromBytes(dtype, data0, 2880, 5760, 32)
					t1f := ctx.(*Context).FromFloatSlice(data0f, 2880, 5760, 32)

					// t.Logf("f32: \n%s", ml.Dump(ctx, t1f))

					t2 := ctx.(*Context).FromFloatSlice(data1[:], 2880, 1, 32)

					t4 := t1.Mulmat(ctx, t2)
					t4f := t1f.Mulmat(ctx, t2)

					d4 := ml.Dump(ctx, t4, ml.DumpWithPrecision(3))
					d4f := ml.Dump(ctx, t4f, ml.DumpWithPrecision(3))

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

				t.Run("exact/3x3", func(t *testing.T) {
					r := rand.New(rand.NewSource(0))
					ctx := initContextOrSkip(t, b, useGPU)
					const s10 = 64
					const s11 = 1
					const s12 = 2
					const s20 = s10
					const s21 = 1
					const s22 = 2

					data := [s10 * s11 * s12]float32{}
					for i := range data {
						data[i] = mxfp4_vals[r.Int()%len(mxfp4_vals)]
					}
					// for i := range len(data) / 32 {
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
					t1 := ctx.(*Context).FromBytes(dtype, mxData, s10, s11, s12)
					t1f := ctx.(*Context).FromFloatSlice(data[:], s10, s11, s12)

					d2 := [s20 * s21 * s22]float32{}
					for i := range d2 {
						d2[i] = float32(r.Float32())
					}
					t2 := ctx.(*Context).FromFloatSlice(d2[:], s20, s21, s22)

					t3f := t1f.Mulmat(ctx, t2)
					t3 := t1.Mulmat(ctx, t2)
					d3 := ml.Dump(ctx, t3)
					d3f := ml.Dump(ctx, t3f)
					if d3 != d3f {
						t.Fatalf("expected (f32): \n%s\n\n but got (mxfp4): \n%s", d3f, d3)
					}
				})

				t.Run("exact/2x2", func(t *testing.T) {
					r := rand.New(rand.NewSource(0))
					ctx := initContextOrSkip(t, b, useGPU)
					const s0 = 32
					const s1 = 64

					data := [s0 * s1]float32{}
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
					t1 := ctx.(*Context).FromBytes(dtype, mxData, s0, s1)
					t1f := ctx.(*Context).FromFloatSlice(data[:], s0, s1)

					d2 := [s0 * s1]float32{}
					for i := range d2 {
						d2[i] = float32(r.Float32())
					}
					t2 := ctx.(*Context).FromFloatSlice(d2[:], s0, s1)

					t3f := t1f.Mulmat(ctx, t2)
					t3 := t1.Mulmat(ctx, t2)
					d3 := ml.Dump(ctx, t3)
					d3f := ml.Dump(ctx, t3f)
					if d3 != d3f {
						t.Fatalf("expected (f32): \n%s\n\n but got (mxfp4): \n%s", d3f, d3)
					}
				})
				t.Run("exact/2x1", func(t *testing.T) {
					r := rand.New(rand.NewSource(0))
					ctx := initContextOrSkip(t, b, useGPU)
					const s0 = 64
					const s1 = 4

					data := [s0 * s1]float32{}
					for i := range data {
						data[i] = mxfp4_vals[r.Int()%len(mxfp4_vals)]
					}
					// for i := range len(data) / 32 {
					// 	vals := [32]string{}
					// 	for j := range vals {
					// 		vals[j] = fmt.Sprintf("%0.2f", data[i*32+j])
					// 	}
					// 	t.Logf("  t1[%s]\n", strings.Join(vals[:], ", "))
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
					t1 := ctx.(*Context).FromBytes(dtype, mxData, s0, s1)
					t1f := ctx.(*Context).FromFloatSlice(data[:], s0, s1)

					d2 := [s0]float32{}
					for i := range d2 {
						d2[i] = float32(r.Float32())
					}
					// for i := range len(d2) / 32 {
					// 	vals := [32]string{}
					// 	for j := range vals {
					// 		vals[j] = fmt.Sprintf("%0.2f", d2[i*32+j])
					// 	}
					// 	t.Logf("  t2[%s]\n", strings.Join(vals[:], ", "))
					// }

					t2 := ctx.(*Context).FromFloatSlice(d2[:], s0)

					t3f := t1f.Mulmat(ctx, t2)
					t3 := t1.Mulmat(ctx, t2)
					d3 := ml.Dump(ctx, t3, ml.DumpWithPrecision(3))
					d3f := ml.Dump(ctx, t3f, ml.DumpWithPrecision(3))
					if d3 != d3f {
						t.Fatalf("expected (f32): \n%s\n\n but got (mxfp4): \n%s", d3f, d3)
					}
				})

				t.Run("range/2d", func(t *testing.T) {
					r := rand.New(rand.NewSource(0))
					ctx := initContextOrSkip(t, b, useGPU)
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
					// for i := range len(data) / 32 {
					// 	vals := [32]string{}
					// 	for j := range vals {
					// 		vals[j] = fmt.Sprintf("%0.2f", dataf[i*32+j])
					// 	}
					// 	t.Logf("  t1[%s]\n", strings.Join(vals[:], ", "))
					// }

					d2 := [s0 * s1]float32{}
					for i := range d2 {
						// d2[i] = float32(i)
						d2[i] = float32(r.Float32())
					}
					// for i := range len(d2) / s0 {
					// 	vals := [s0]string{}
					// 	for j := range vals {
					// 		vals[j] = fmt.Sprintf("%0.2f", d2[i*s0+j])
					// 	}
					// 	t.Logf("  t2[%s]\n", strings.Join(vals[:], ", "))
					// }

					t2 := ctx.(*Context).FromFloatSlice(d2[:], s0, s1)

					// t.Log("calling Mulmat")
					t3 := t1.Mulmat(ctx, t2)
					t3f := t1f.Mulmat(ctx, t2)
					d3 := ml.Dump(ctx, t3, ml.DumpWithPrecision(2))
					d3f := ml.Dump(ctx, t3f, ml.DumpWithPrecision(2))
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
					ctx := initContextOrSkip(t, b, useGPU)
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

					// t.Log("calling Mulmat")
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
		})
	}
}

func TestMXFP4Simple(t *testing.T) {
	b := setup(t)

	t.Run("fixed", func(t *testing.T) {
		ctx := initContextOrSkip(t, b, false)
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
			data[i] = mxfp4_vals[r.Int()%len(mxfp4_vals)]
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
