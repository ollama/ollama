package ggml

import (
	"errors"
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
