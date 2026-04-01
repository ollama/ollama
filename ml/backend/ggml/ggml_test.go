package ggml

import (
	"errors"
	"fmt"
	"os"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
)

func setup(tb testing.TB) ml.Context {
	tb.Helper()

	f, err := os.CreateTemp(tb.TempDir(), "*.bin")
	if err != nil {
		tb.Fatal(err)
	}
	defer f.Close()

	if err := ggml.WriteGGUF(f, ggml.KV{"general.architecture": "test"}, nil); err != nil {
		tb.Fatal(err)
	}

	b, err := ml.NewBackend(f.Name(), ml.BackendParams{AllocMemory: true})
	if err != nil {
		tb.Fatal(err)
	}

	ctx := b.NewContext().Input()

	tb.Cleanup(func() {
		ctx.Close()
		b.Close()
	})

	return ctx
}

func TestInferShape(t *testing.T) {
	cases := []struct {
		name  string
		input []int
		want  []int
		err   error
	}{
		{
			name:  "no inferred shape",
			input: []int{2, 3, 4},
			want:  []int{2, 3, 4},
		},
		{
			name:  "infer begin",
			input: []int{-1, 3, 4},
			want:  []int{2, 3, 4},
		},
		{
			name:  "infer mid",
			input: []int{2, -1, 4},
			want:  []int{2, 3, 4},
		},
		{
			name:  "infer end",
			input: []int{2, 3, -1},
			want:  []int{2, 3, 4},
		},
		{
			name:  "too many inferred dims",
			input: []int{-1, 3, -1},
			err:   errors.New("only one dimension can be inferred"),
		},
		{
			name:  "infer gather",
			input: []int{2, -1},
			want:  []int{2, 12},
		},
		{
			name:  "infer gather all",
			input: []int{-1},
			want:  []int{24},
		},
		{
			name:  "infer split",
			input: []int{2, -1, 3, 2},
			want:  []int{2, 2, 3, 2},
		},
		{
			name:  "indivisible infer",
			input: []int{2, -1, 2, 4},
			err:   errors.New("cannot infer dimension"),
		},
		{
			name:  "infer zero dim",
			input: []int{2, 0, 4},
			err:   errors.New("dimension cannot be zero"),
		},
	}

	ctx := setup(t)
	tensor, ok := ctx.Empty(ml.DTypeF32, 2, 3, 4).(*Tensor)
	if !ok {
		t.Fatal("expected *Tensor")
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil && tt.err == nil {
					// all good
				} else if r != nil && tt.err == nil {
					t.Errorf("unexpected panic: %v", r)
				} else if r == nil && tt.err != nil {
					t.Errorf("expected panic but did not get one: %v", tt.err)
				} else if errStr, ok := r.(string); ok && errStr != tt.err.Error() {
					t.Errorf("expected panic %q but got %q", tt.err.Error(), errStr)
				}
			}()

			inferShape(tensor, tt.input)
			if diff := cmp.Diff(tt.want, tt.input); diff != "" {
				t.Errorf("%s: shape mismatch (-want +got):\n%s", tt.name, diff)
			}
		})
	}
}

func EquateTensors(ctx ml.Context) cmp.Option {
	return cmp.Comparer(func(x, y ml.Tensor) bool {
		ctx.Forward(x, y).Compute(x, y)
		return cmp.Equal(x.Shape(), y.Shape()) &&
			cmp.Equal(x.DType(), y.DType()) &&
			cmp.Equal(x.Bytes(), y.Bytes())
	})
}

func TestMulmat(t *testing.T) {
	cases := []struct {
		name    string
		a, b, c func(ml.Context) ml.Tensor
	}{
		{
			name: "vector x vector",
			a: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 3, 1, ml.DTypeF32)
			},
			b: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 3, 1, ml.DTypeF32)
			},
			c: func(ctx ml.Context) ml.Tensor {
				return ctx.FromFloats([]float32{5}, 1)
			},
		},
		{
			name: "vector x matrix",
			a: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 4, 1, ml.DTypeF32)
			},
			b: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 12, 1, ml.DTypeF32).Reshape(ctx, 4, 3)
			},
			c: func(ctx ml.Context) ml.Tensor {
				return ctx.FromFloats([]float32{
					14, 38, 62,
				}, 1, 3)
			},
		},
		{
			name: "broadcast vector x batched matrix",
			a: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 4, 1, ml.DTypeF32)
			},
			b: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 10*3*4, 1, ml.DTypeF32).Reshape(ctx, 4, 3, 10)
			},
			c: func(ctx ml.Context) ml.Tensor {
				return ctx.FromFloats([]float32{
					14, 38, 62,
					86, 110, 134,
					158, 182, 206,
					230, 254, 278,
					302, 326, 350,
					374, 398, 422,
					446, 470, 494,
					518, 542, 566,
					590, 614, 638,
					662, 686, 710,
				}, 1, 3, 10)
			},
		},
		{
			name: "batched matrix x batched matrix",
			a: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 4*5*10, 1, ml.DTypeF32).Reshape(ctx, 4, 5, 10)
			},
			b: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 4*3*10, 1, ml.DTypeF32).Reshape(ctx, 4, 3, 10)
			},
			c: func(ctx ml.Context) ml.Tensor {
				return ctx.FromFloats([]float32{
					14, 38, 62, 86, 110,
					38, 126, 214, 302, 390,
					62, 214, 366, 518, 670,

					1166, 1382, 1598, 1814, 2030,
					1510, 1790, 2070, 2350, 2630,
					1854, 2198, 2542, 2886, 3230,

					4238, 4646, 5054, 5462, 5870,
					4902, 5374, 5846, 6318, 6790,
					5566, 6102, 6638, 7174, 7710,

					9230, 9830, 10430, 11030, 11630,
					10214, 10878, 11542, 12206, 12870,
					11198, 11926, 12654, 13382, 14110,

					16142, 16934, 17726, 18518, 19310,
					17446, 18302, 19158, 20014, 20870,
					18750, 19670, 20590, 21510, 22430,

					24974, 25958, 26942, 27926, 28910,
					26598, 27646, 28694, 29742, 30790,
					28222, 29334, 30446, 31558, 32670,

					35726, 36902, 38078, 39254, 40430,
					37670, 38910, 40150, 41390, 42630,
					39614, 40918, 42222, 43526, 44830,

					48398, 49766, 51134, 52502, 53870,
					50662, 52094, 53526, 54958, 56390,
					52926, 54422, 55918, 57414, 58910,

					62990, 64550, 66110, 67670, 69230,
					65574, 67198, 68822, 70446, 72070,
					68158, 69846, 71534, 73222, 74910,

					79502, 81254, 83006, 84758, 86510,
					82406, 84222, 86038, 87854, 89670,
					85310, 87190, 89070, 90950, 92830,
				}, 5, 3, 10)
			},
		},
		{
			name: "broadcast matrix x batched matrix",
			a: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 4*5, 1, ml.DTypeF32).Reshape(ctx, 4, 5)
			},
			b: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 4*3*10, 1, ml.DTypeF32).Reshape(ctx, 4, 3, 10)
			},
			c: func(ctx ml.Context) ml.Tensor {
				return ctx.FromFloats([]float32{
					14, 38, 62, 86, 110,
					38, 126, 214, 302, 390,
					62, 214, 366, 518, 670,

					86, 302, 518, 734, 950,
					110, 390, 670, 950, 1230,
					134, 478, 822, 1166, 1510,

					158, 566, 974, 1382, 1790,
					182, 654, 1126, 1598, 2070,
					206, 742, 1278, 1814, 2350,

					230, 830, 1430, 2030, 2630,
					254, 918, 1582, 2246, 2910,
					278, 1006, 1734, 2462, 3190,

					302, 1094, 1886, 2678, 3470,
					326, 1182, 2038, 2894, 3750,
					350, 1270, 2190, 3110, 4030,

					374, 1358, 2342, 3326, 4310,
					398, 1446, 2494, 3542, 4590,
					422, 1534, 2646, 3758, 4870,

					446, 1622, 2798, 3974, 5150,
					470, 1710, 2950, 4190, 5430,
					494, 1798, 3102, 4406, 5710,

					518, 1886, 3254, 4622, 5990,
					542, 1974, 3406, 4838, 6270,
					566, 2062, 3558, 5054, 6550,

					590, 2150, 3710, 5270, 6830,
					614, 2238, 3862, 5486, 7110,
					638, 2326, 4014, 5702, 7390,

					662, 2414, 4166, 5918, 7670,
					686, 2502, 4318, 6134, 7950,
					710, 2590, 4470, 6350, 8230,
				}, 5, 3, 10)
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			ctx := setup(t)
			a, b := tt.a(ctx), tt.b(ctx)
			c := a.Mulmat(ctx, b)
			if diff := cmp.Diff(tt.c(ctx), c, EquateTensors(ctx)); diff != "" {
				t.Errorf("MulMat() result mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestPermute(t *testing.T) {
	cases := []struct {
		name  string
		input func(ml.Context) ml.Tensor
		shape []int
		want  func(ml.Context) ml.Tensor
	}{
		{
			name: "transpose",
			input: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 3*2, 1, ml.DTypeF32).Reshape(ctx, 3, 2)
			},
			shape: []int{1, 0, 2, 3},
			want: func(ctx ml.Context) ml.Tensor {
				return ctx.FromFloats([]float32{
					0, 3,
					1, 4,
					2, 5,
				}, 2, 3)
			},
		},
		{
			name: "transpose fill dims",
			input: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 3*2, 1, ml.DTypeF32).Reshape(ctx, 3, 2)
			},
			shape: []int{1, 0},
			want: func(ctx ml.Context) ml.Tensor {
				return ctx.FromFloats([]float32{
					0, 3,
					1, 4,
					2, 5,
				}, 2, 3)
			},
		},
		{
			name: "permute 3d",
			input: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 5*3*2, 1, ml.DTypeF32).Reshape(ctx, 2, 3, 5)
			},
			shape: []int{2, 0, 1, 3},
			want: func(ctx ml.Context) ml.Tensor {
				return ctx.FromFloats([]float32{
					0, 2, 4,
					6, 8, 10,
					12, 14, 16,
					18, 20, 22,
					24, 26, 28,

					1, 3, 5,
					7, 9, 11,
					13, 15, 17,
					19, 21, 23,
					25, 27, 29,
				}, 3, 5, 2)
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			ctx := setup(t)
			got := tt.input(ctx).Permute(ctx, tt.shape...)
			got = got.Contiguous(ctx)
			if diff := cmp.Diff(tt.want(ctx), got, EquateTensors(ctx)); diff != "" {
				t.Errorf("Permute() result mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSlice(t *testing.T) {
	cases := []struct {
		dim   int
		low   int
		high  int
		step  int
		input func(ml.Context) ml.Tensor
		want  func(ml.Context) ml.Tensor
	}{
		{
			dim: 0, low: 1, high: 3, step: 1,
			input: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 4*4*4*4, 1, ml.DTypeF32).Reshape(ctx, 4, 4, 4, 4)
			},
			want: func(ctx ml.Context) ml.Tensor {
				return ctx.FromFloats([]float32{
					1, 2,
					5, 6,
					9, 10,
					13, 14,

					17, 18,
					21, 22,
					25, 26,
					29, 30,

					33, 34,
					37, 38,
					41, 42,
					45, 46,

					49, 50,
					53, 54,
					57, 58,
					61, 62,

					65, 66,
					69, 70,
					73, 74,
					77, 78,

					81, 82,
					85, 86,
					89, 90,
					93, 94,

					97, 98,
					101, 102,
					105, 106,
					109, 110,

					113, 114,
					117, 118,
					121, 122,
					125, 126,

					129, 130,
					133, 134,
					137, 138,
					141, 142,

					145, 146,
					149, 150,
					153, 154,
					157, 158,

					161, 162,
					165, 166,
					169, 170,
					173, 174,

					177, 178,
					181, 182,
					185, 186,
					189, 190,

					193, 194,
					197, 198,
					201, 202,
					205, 206,

					209, 210,
					213, 214,
					217, 218,
					221, 222,

					225, 226,
					229, 230,
					233, 234,
					237, 238,

					241, 242,
					245, 246,
					249, 250,
					253, 254,
				}, 2, 4, 4, 4)
			},
		},
		{
			dim: 1, low: 1, high: 3, step: 1,
			input: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 4*4*4*4, 1, ml.DTypeF32).Reshape(ctx, 4, 4, 4, 4)
			},
			want: func(ctx ml.Context) ml.Tensor {
				return ctx.FromFloats([]float32{
					4, 5, 6, 7,
					8, 9, 10, 11,

					20, 21, 22, 23,
					24, 25, 26, 27,

					36, 37, 38, 39,
					40, 41, 42, 43,

					52, 53, 54, 55,
					56, 57, 58, 59,

					68, 69, 70, 71,
					72, 73, 74, 75,

					84, 85, 86, 87,
					88, 89, 90, 91,

					100, 101, 102, 103,
					104, 105, 106, 107,

					116, 117, 118, 119,
					120, 121, 122, 123,

					132, 133, 134, 135,
					136, 137, 138, 139,

					148, 149, 150, 151,
					152, 153, 154, 155,

					164, 165, 166, 167,
					168, 169, 170, 171,

					180, 181, 182, 183,
					184, 185, 186, 187,

					196, 197, 198, 199,
					200, 201, 202, 203,

					212, 213, 214, 215,
					216, 217, 218, 219,

					228, 229, 230, 231,
					232, 233, 234, 235,

					244, 245, 246, 247,
					248, 249, 250, 251,
				}, 4, 2, 4, 4)
			},
		},
		{
			dim: 2, low: 1, high: 3, step: 1,
			input: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 4*4*4*4, 1, ml.DTypeF32).Reshape(ctx, 4, 4, 4, 4)
			},
			want: func(ctx ml.Context) ml.Tensor {
				return ctx.FromFloats([]float32{
					16, 17, 18, 19,
					20, 21, 22, 23,
					24, 25, 26, 27,
					28, 29, 30, 31,

					32, 33, 34, 35,
					36, 37, 38, 39,
					40, 41, 42, 43,
					44, 45, 46, 47,

					80, 81, 82, 83,
					84, 85, 86, 87,
					88, 89, 90, 91,
					92, 93, 94, 95,

					96, 97, 98, 99,
					100, 101, 102, 103,
					104, 105, 106, 107,
					108, 109, 110, 111,

					144, 145, 146, 147,
					148, 149, 150, 151,
					152, 153, 154, 155,
					156, 157, 158, 159,

					160, 161, 162, 163,
					164, 165, 166, 167,
					168, 169, 170, 171,
					172, 173, 174, 175,

					208, 209, 210, 211,
					212, 213, 214, 215,
					216, 217, 218, 219,
					220, 221, 222, 223,

					224, 225, 226, 227,
					228, 229, 230, 231,
					232, 233, 234, 235,
					236, 237, 238, 239,
				}, 4, 4, 2, 4)
			},
		},
		{
			dim: 3, low: 1, high: 3, step: 1,
			input: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 4*4*4*4, 1, ml.DTypeF32).Reshape(ctx, 4, 4, 4, 4)
			},
			want: func(ctx ml.Context) ml.Tensor {
				return ctx.FromFloats([]float32{
					64, 65, 66, 67,
					68, 69, 70, 71,
					72, 73, 74, 75,
					76, 77, 78, 79,

					80, 81, 82, 83,
					84, 85, 86, 87,
					88, 89, 90, 91,
					92, 93, 94, 95,

					96, 97, 98, 99,
					100, 101, 102, 103,
					104, 105, 106, 107,
					108, 109, 110, 111,

					112, 113, 114, 115,
					116, 117, 118, 119,
					120, 121, 122, 123,
					124, 125, 126, 127,

					128, 129, 130, 131,
					132, 133, 134, 135,
					136, 137, 138, 139,
					140, 141, 142, 143,

					144, 145, 146, 147,
					148, 149, 150, 151,
					152, 153, 154, 155,
					156, 157, 158, 159,

					160, 161, 162, 163,
					164, 165, 166, 167,
					168, 169, 170, 171,
					172, 173, 174, 175,

					176, 177, 178, 179,
					180, 181, 182, 183,
					184, 185, 186, 187,
					188, 189, 190, 191,
				}, 4, 4, 4, 2)
			},
		},
		{
			dim: 0, low: 0, high: 4, step: 2,
			input: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 4*4*4*4, 1, ml.DTypeF32).Reshape(ctx, 4, 4, 4, 4)
			},
			want: func(ctx ml.Context) ml.Tensor {
				return ctx.FromFloats([]float32{
					0, 2,
					4, 6,
					8, 10,
					12, 14,

					16, 18,
					20, 22,
					24, 26,
					28, 30,

					32, 34,
					36, 38,
					40, 42,
					44, 46,

					48, 50,
					52, 54,
					56, 58,
					60, 62,

					64, 66,
					68, 70,
					72, 74,
					76, 78,

					80, 82,
					84, 86,
					88, 90,
					92, 94,

					96, 98,
					100, 102,
					104, 106,
					108, 110,

					112, 114,
					116, 118,
					120, 122,
					124, 126,

					128, 130,
					132, 134,
					136, 138,
					140, 142,

					144, 146,
					148, 150,
					152, 154,
					156, 158,

					160, 162,
					164, 166,
					168, 170,
					172, 174,

					176, 178,
					180, 182,
					184, 186,
					188, 190,

					192, 194,
					196, 198,
					200, 202,
					204, 206,

					208, 210,
					212, 214,
					216, 218,
					220, 222,

					224, 226,
					228, 230,
					232, 234,
					236, 238,

					240, 242,
					244, 246,
					248, 250,
					252, 254,
				}, 2, 4, 4, 4)
			},
		},
		{
			dim: 1, low: 0, high: 4, step: 2,
			input: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 4*4*4*4, 1, ml.DTypeF32).Reshape(ctx, 4, 4, 4, 4)
			},
			want: func(ctx ml.Context) ml.Tensor {
				return ctx.FromFloats([]float32{
					0, 1, 2, 3,
					8, 9, 10, 11,

					16, 17, 18, 19,
					24, 25, 26, 27,

					32, 33, 34, 35,
					40, 41, 42, 43,

					48, 49, 50, 51,
					56, 57, 58, 59,

					64, 65, 66, 67,
					72, 73, 74, 75,

					80, 81, 82, 83,
					88, 89, 90, 91,

					96, 97, 98, 99,
					104, 105, 106, 107,

					112, 113, 114, 115,
					120, 121, 122, 123,

					128, 129, 130, 131,
					136, 137, 138, 139,

					144, 145, 146, 147,
					152, 153, 154, 155,

					160, 161, 162, 163,
					168, 169, 170, 171,

					176, 177, 178, 179,
					184, 185, 186, 187,

					192, 193, 194, 195,
					200, 201, 202, 203,

					208, 209, 210, 211,
					216, 217, 218, 219,

					224, 225, 226, 227,
					232, 233, 234, 235,

					240, 241, 242, 243,
					248, 249, 250, 251,
				}, 4, 2, 4, 4)
			},
		},
		{
			dim: 2, low: 0, high: 4, step: 2,
			input: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 4*4*4*4, 1, ml.DTypeF32).Reshape(ctx, 4, 4, 4, 4)
			},
			want: func(ctx ml.Context) ml.Tensor {
				return ctx.FromFloats([]float32{
					0, 1, 2, 3,
					4, 5, 6, 7,
					8, 9, 10, 11,
					12, 13, 14, 15,

					32, 33, 34, 35,
					36, 37, 38, 39,
					40, 41, 42, 43,
					44, 45, 46, 47,

					64, 65, 66, 67,
					68, 69, 70, 71,
					72, 73, 74, 75,
					76, 77, 78, 79,

					96, 97, 98, 99,
					100, 101, 102, 103,
					104, 105, 106, 107,
					108, 109, 110, 111,

					128, 129, 130, 131,
					132, 133, 134, 135,
					136, 137, 138, 139,
					140, 141, 142, 143,

					160, 161, 162, 163,
					164, 165, 166, 167,
					168, 169, 170, 171,
					172, 173, 174, 175,

					192, 193, 194, 195,
					196, 197, 198, 199,
					200, 201, 202, 203,
					204, 205, 206, 207,

					224, 225, 226, 227,
					228, 229, 230, 231,
					232, 233, 234, 235,
					236, 237, 238, 239,
				}, 4, 4, 2, 4)
			},
		},
		{
			dim: 3, low: 0, high: 4, step: 2,
			input: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 4*4*4*4, 1, ml.DTypeF32).Reshape(ctx, 4, 4, 4, 4)
			},
			want: func(ctx ml.Context) ml.Tensor {
				return ctx.FromFloats([]float32{
					0, 1, 2, 3,
					4, 5, 6, 7,
					8, 9, 10, 11,
					12, 13, 14, 15,

					16, 17, 18, 19,
					20, 21, 22, 23,
					24, 25, 26, 27,
					28, 29, 30, 31,

					32, 33, 34, 35,
					36, 37, 38, 39,
					40, 41, 42, 43,
					44, 45, 46, 47,

					48, 49, 50, 51,
					52, 53, 54, 55,
					56, 57, 58, 59,
					60, 61, 62, 63,

					128, 129, 130, 131,
					132, 133, 134, 135,
					136, 137, 138, 139,
					140, 141, 142, 143,

					144, 145, 146, 147,
					148, 149, 150, 151,
					152, 153, 154, 155,
					156, 157, 158, 159,

					160, 161, 162, 163,
					164, 165, 166, 167,
					168, 169, 170, 171,
					172, 173, 174, 175,

					176, 177, 178, 179,
					180, 181, 182, 183,
					184, 185, 186, 187,
					188, 189, 190, 191,
				}, 4, 4, 4, 2)
			},
		},
	}

	for _, tt := range cases {
		name := fmt.Sprintf("dim=%d,low=%d,high=%d,step=%d", tt.dim, tt.low, tt.high, tt.step)
		t.Run(name, func(t *testing.T) {
			ctx := setup(t)
			got := tt.input(ctx).Slice(ctx, tt.dim, tt.low, tt.high, tt.step)
			got = got.Contiguous(ctx)
			if diff := cmp.Diff(tt.want(ctx), got, EquateTensors(ctx)); diff != "" {
				t.Errorf("Slice() result mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSplitSections(t *testing.T) {
	cases := []struct {
		dim      int
		sections []int
		input    func(ml.Context) ml.Tensor
		want     []func(ml.Context) ml.Tensor
	}{
		{
			dim: 0, sections: []int{1, 1, 1},
			input: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 12, 1, ml.DTypeF32).Reshape(ctx, 3, 4)
			},
			want: []func(ml.Context) ml.Tensor{
				func(ctx ml.Context) ml.Tensor {
					return ctx.FromFloats([]float32{0, 3, 6, 9}, 1, 4)
				},
				func(ctx ml.Context) ml.Tensor {
					return ctx.FromFloats([]float32{1, 4, 7, 10}, 1, 4)
				},
				func(ctx ml.Context) ml.Tensor {
					return ctx.FromFloats([]float32{2, 5, 8, 11}, 1, 4)
				},
			},
		},
		{
			dim: 1, sections: []int{1, 3},
			input: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 12, 1, ml.DTypeF32).Reshape(ctx, 3, 4)
			},
			want: []func(ml.Context) ml.Tensor{
				func(ctx ml.Context) ml.Tensor {
					return ctx.FromFloats([]float32{0, 1, 2}, 3, 1)
				},
				func(ctx ml.Context) ml.Tensor {
					return ctx.FromFloats([]float32{
						3, 4, 5,
						6, 7, 8,
						9, 10, 11,
					}, 3, 3)
				},
			},
		},
		{
			dim: 0, sections: []int{2, 2},
			input: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 12, 1, ml.DTypeF32).Reshape(ctx, 4, 3)
			},
			want: []func(ml.Context) ml.Tensor{
				func(ctx ml.Context) ml.Tensor {
					return ctx.FromFloats([]float32{
						0, 1,
						4, 5,
						8, 9,
					}, 2, 3)
				},
				func(ctx ml.Context) ml.Tensor {
					return ctx.FromFloats([]float32{
						2, 3,
						6, 7,
						10, 11,
					}, 2, 3)
				},
			},
		},
		{
			dim: 1, sections: []int{1, 2},
			input: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 12, 1, ml.DTypeF32).Reshape(ctx, 4, 3)
			},
			want: []func(ml.Context) ml.Tensor{
				func(ctx ml.Context) ml.Tensor {
					return ctx.FromFloats([]float32{0, 1, 2, 3}, 4, 1)
				},
				func(ctx ml.Context) ml.Tensor {
					return ctx.FromFloats([]float32{
						4, 5, 6, 7,
						8, 9, 10, 11,
					}, 4, 2)
				},
			},
		},
	}

	for _, tt := range cases {
		t.Run(fmt.Sprintf("sections=%v", tt.sections), func(t *testing.T) {
			ctx := setup(t)
			got := tt.input(ctx).ChunkSections(ctx, tt.dim, tt.sections...)

			for i := range got {
				got[i] = got[i].Contiguous(ctx)
			}

			ctx.Forward(got...).Compute(got...)
			for i, want := range tt.want {
				if diff := cmp.Diff(want(ctx), got[i], EquateTensors(ctx)); diff != "" {
					t.Errorf("SplitSections() section %d mismatch (-want +got):\n%s", i, diff)
				}
			}
		})
	}
}

func TestChunk(t *testing.T) {
	cases := []struct {
		dim   int
		chunk int
		input func(ml.Context) ml.Tensor
		want  []func(ml.Context) ml.Tensor
	}{
		{
			dim: 0, chunk: 1,
			input: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 12, 1, ml.DTypeF32).Reshape(ctx, 3, 4)
			},
			want: []func(ml.Context) ml.Tensor{
				func(ctx ml.Context) ml.Tensor {
					return ctx.FromFloats([]float32{0, 3, 6, 9}, 1, 4)
				},
				func(ctx ml.Context) ml.Tensor {
					return ctx.FromFloats([]float32{1, 4, 7, 10}, 1, 4)
				},
				func(ctx ml.Context) ml.Tensor {
					return ctx.FromFloats([]float32{2, 5, 8, 11}, 1, 4)
				},
			},
		},
		{
			dim: 1, chunk: 2,
			input: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 12, 1, ml.DTypeF32).Reshape(ctx, 3, 4)
			},
			want: []func(ml.Context) ml.Tensor{
				func(ctx ml.Context) ml.Tensor {
					return ctx.FromFloats([]float32{
						0, 1, 2,
						3, 4, 5,
					}, 3, 2)
				},
				func(ctx ml.Context) ml.Tensor {
					return ctx.FromFloats([]float32{
						6, 7, 8,
						9, 10, 11,
					}, 3, 2)
				},
			},
		},
		{
			dim: 0, chunk: 2,
			input: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 12, 1, ml.DTypeF32).Reshape(ctx, 3, 4)
			},
			want: []func(ml.Context) ml.Tensor{
				func(ctx ml.Context) ml.Tensor {
					return ctx.FromFloats([]float32{
						0, 1,
						3, 4,
						6, 7,
						9, 10,
					}, 2, 4)
				},
				func(ctx ml.Context) ml.Tensor {
					return ctx.FromFloats([]float32{
						2,
						5,
						8,
						11,
					}, 1, 4)
				},
			},
		},
	}

	for _, tt := range cases {
		t.Run(fmt.Sprintf("dim=%d,chunk=%d", tt.dim, tt.chunk), func(t *testing.T) {
			ctx := setup(t)
			got := tt.input(ctx).Chunk(ctx, tt.dim, tt.chunk)

			for i := range got {
				got[i] = got[i].Contiguous(ctx)
			}

			ctx.Forward(got...).Compute(got...)
			for i, want := range tt.want {
				if diff := cmp.Diff(want(ctx), got[i], EquateTensors(ctx)); diff != "" {
					t.Errorf("Split() section %d mismatch (-want +got):\n%s", i, diff)
				}
			}
		})
	}
}
