package ggml

import (
	"fmt"
	"log/slog"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestPermute2d(t *testing.T) {
	be := newTestBackend(100)
	ctx := newTestContext(be, 100).Input()
	shape := []int{2, 3}
	data := make([]float32, shape[0]*shape[1])
	for i := range data {
		data[i] = float32(i + 1)
	}
	x, err := ctx.FromFloatSlice(data, shape...)
	if err != nil {
		t.Fatal(err)
	}
	slog.Info("Initial data", "tensor", x)

	type testCase struct {
		shape []int
	}
	testCases := []testCase{
		{shape: []int{0, 1, 2, 3}},
		{shape: []int{1, 0, 2, 3}},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%v", tc), func(t *testing.T) {
			t2 := x.Permute(ctx, tc.shape...)
			slog.Info("After permute", "request", tc.shape, "result", t2)
			res := t2.Shape()
			expected := [2]int{
				shape[tc.shape[0]],
				shape[tc.shape[1]],
			}
			if ([2]int)(res) != expected {
				t.Fatalf("reshape %v expected %v but got %v", tc.shape, expected, res)
			}
		})
	}
}

func TestPermute3d(t *testing.T) {
	be := newTestBackend(100)
	ctx := newTestContext(be, 100).Input()
	shape := []int{2, 3, 5}
	data := make([]float32, shape[0]*shape[1]*shape[2])
	for i := range data {
		data[i] = float32(i + 1)
	}
	x, err := ctx.FromFloatSlice(data, shape...)
	if err != nil {
		t.Fatal(err)
	}
	slog.Info("Initial data", "tensor", x)

	type testCase struct {
		shape []int
	}
	testCases := []testCase{
		{shape: []int{0, 1, 2, 3}},
		{shape: []int{0, 2, 1, 3}},
		{shape: []int{1, 0, 2, 3}},
		{shape: []int{1, 2, 0, 3}},
		{shape: []int{2, 0, 1, 3}},
		{shape: []int{2, 1, 0, 3}},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%v", tc), func(t *testing.T) {
			t2 := x.Permute(ctx, tc.shape...)
			slog.Info("After permute", "request", tc.shape, "result", t2)
			res := t2.Shape()
			expected := [3]int{
				shape[tc.shape[0]],
				shape[tc.shape[1]],
				shape[tc.shape[2]],
			}
			if ([3]int)(res) != expected {
				t.Fatalf("reshape %v expected %v but got %v", tc.shape, expected, res)
			}
		})
	}
}

func TestPermute4d(t *testing.T) {
	be := newTestBackend(1000)
	ctx := newTestContext(be, 1000).Input()
	shape := []int{2, 3, 5, 7}
	data := make([]float32, shape[0]*shape[1]*shape[2]*shape[3])
	for i := range data {
		data[i] = float32(i + 1)
	}
	x, err := ctx.FromFloatSlice(data, shape...)
	if err != nil {
		t.Fatal(err)
	}
	slog.Info("Initial data", "tensor", x)

	type testCase struct {
		shape []int
	}
	testCases := []testCase{
		{shape: []int{0, 1, 2, 3}},
		{shape: []int{0, 1, 3, 2}},
		{shape: []int{0, 2, 1, 3}},
		{shape: []int{0, 2, 3, 1}},
		{shape: []int{0, 3, 1, 2}},
		{shape: []int{0, 3, 2, 1}},
		{shape: []int{1, 0, 2, 3}},
		{shape: []int{1, 0, 3, 2}},
		{shape: []int{1, 2, 0, 3}},
		{shape: []int{1, 2, 3, 0}},
		{shape: []int{1, 3, 0, 2}},
		{shape: []int{1, 3, 2, 0}},
		{shape: []int{2, 0, 1, 3}},
		{shape: []int{2, 0, 3, 1}},
		{shape: []int{2, 1, 0, 3}},
		{shape: []int{2, 1, 3, 0}},
		{shape: []int{2, 3, 0, 1}},
		{shape: []int{2, 3, 1, 0}},
		{shape: []int{3, 0, 1, 2}},
		{shape: []int{3, 0, 2, 1}},
		{shape: []int{3, 1, 0, 2}},
		{shape: []int{3, 1, 2, 0}},
		{shape: []int{3, 2, 0, 1}},
		{shape: []int{3, 2, 1, 0}},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%v", tc), func(t *testing.T) {
			t2 := x.Permute(ctx, tc.shape...)
			slog.Info("After permute", "request", tc.shape, "result", t2)
			res := t2.Shape()
			expected := [4]int{
				shape[tc.shape[0]],
				shape[tc.shape[1]],
				shape[tc.shape[2]],
				shape[tc.shape[3]],
			}
			if ([4]int)(res) != expected {
				t.Fatalf("reshape %v expected %v but got %v", tc.shape, expected, res)
			}
		})
	}
}

func arange(t *testing.T, ctx ml.Context, shape []int) ml.Tensor {
	size := 1
	for _, d := range shape {
		size *= d
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(i)
	}
	tensor, err := ctx.FromFloatSlice(data, shape...)
	if err != nil {
		t.Fatal(err)
	}
	return tensor
}

func TestRowOrderMul(t *testing.T) {
	/* Equivalent to the following pytorch

	import torch
	t1 = torch.arange(12).view(3,4)
	print(t1)
	t2 = torch.arange(4).view(1,4)
	print(t2)
	t3 = torch.mul(t1,t2)
	print(t3)
	*/

	type testCase struct {
		s1      []int
		s2      []int
		expVals []float32
	}
	testCases := []testCase{
		{
			s1: []int{4},
			s2: []int{1},
			expVals: []float32{
				0, 0, 0, 0,
			},
		},
		{
			s1: []int{4},
			s2: []int{4},
			expVals: []float32{
				0, 1, 4, 9,
			},
		},
		{
			s1: []int{3, 4},
			s2: []int{1, 4},
			expVals: []float32{
				0, 1, 4, 9,
				0, 5, 12, 21,
				0, 9, 20, 33,
			},
		},
		{
			s1: []int{2, 3, 4},
			s2: []int{1, 4},
			expVals: []float32{
				0, 1, 4, 9,
				0, 5, 12, 21,
				0, 9, 20, 33,

				0, 13, 28, 45,
				0, 17, 36, 57,
				0, 21, 44, 69,
			},
		},
		{
			s1: []int{2, 2, 3, 4},
			s2: []int{1, 4},
			expVals: []float32{
				0, 1, 4, 9,
				0, 5, 12, 21,
				0, 9, 20, 33,

				0, 13, 28, 45,
				0, 17, 36, 57,
				0, 21, 44, 69,

				0, 25, 52, 81,
				0, 29, 60, 93,
				0, 33, 68, 105,

				0, 37, 76, 117,
				0, 41, 84, 129,
				0, 45, 92, 141,
			},
		},
	}

	be := newTestBackend(100)
	ctx := newTestContext(be, 100).Input()

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%v_%v", tc.s1, tc.s2), func(t *testing.T) {

			t1 := arange(t, ctx, tc.s1)
			slog.Info("Created", "t1", t1)
			ctx.Forward(t1)
			fmt.Fprintln(os.Stderr, ml.Dump(ctx, t1))
			t2 := arange(t, ctx, tc.s2)
			slog.Info("Created", "t2", t2)
			ctx.Forward(t2)
			fmt.Fprintln(os.Stderr, ml.Dump(ctx, t2))
			t3 := t1.Mul(ctx, t2)
			ctx.Forward(t3)
			fmt.Fprintf(os.Stderr, "Dumping t1 * t2 result\n")
			fmt.Fprintln(os.Stderr, ml.Dump(ctx, t3))
			res := t3.Floats()
			if !reflect.DeepEqual(res, tc.expVals) {
				t.Fatalf("incorrect result\ngot %v\nexpected %v", res, tc.expVals)
			}
		})
	}
}

func TestRowOrderMatmul(t *testing.T) {
	/* Equivalent to the following pytorch - all expValues and expShapes derived from this

	import torch
	t1 = torch.arange(12).view(3,4)
	print(t1)
	t2 = torch.arange(4).view(1,4)
	print(t2)
	t3 = torch.matmul(t1,t2)
	print(t3, t3.shape)
	*/

	type testCase struct {
		s1       []int
		s2       []int
		expVals  []float32
		expShape []int
		expError string
	}
	testCases := []testCase{
		// If both tensors are 1-dimensional, the dot product (scalar) is returned.
		{
			s1: []int{4},
			s2: []int{4},
			expVals: []float32{
				14,
			},
			expShape: []int{1},
		},
		{
			s1:       []int{4},
			s2:       []int{2},
			expError: "malformed tensors",
		},
		{
			s1:       []int{2},
			s2:       []int{4},
			expError: "malformed tensors",
		},

		// If both arguments are 2-dimensional, the matrix-matrix product is returned.
		{
			s1: []int{4, 2},
			s2: []int{2, 4},
			expVals: []float32{
				4, 5, 6, 7,
				12, 17, 22, 27,
				20, 29, 38, 47,
				28, 41, 54, 67,
			},
			expShape: []int{4, 4},
		},
		{
			s1: []int{2, 4},
			s2: []int{4, 3},
			expVals: []float32{
				42, 48, 54,
				114, 136, 158,
			},
			expShape: []int{2, 3},
		},
		{
			s1: []int{4, 4},
			s2: []int{4, 2},
			expVals: []float32{
				28, 34,
				76, 98,
				124, 162,
				172, 226,
			},
			expShape: []int{4, 2},
		},
		{
			s1:       []int{4, 2},
			s2:       []int{4, 2},
			expError: "malformed tensors",
		},
		{
			s1:       []int{4, 2},
			s2:       []int{4, 4},
			expError: "malformed tensors",
		},

		// If the first argument is 1-dimensional and the second argument is
		// 2-dimensional, a 1 is prepended to its dimension for the purpose of
		// the matrix multiply. After the matrix multiply, the prepended
		// dimension is removed.
		{
			s1: []int{4},
			s2: []int{4, 2},
			expVals: []float32{
				28, 34,
			},
			expShape: []int{2},
		},
		{
			s1:       []int{3},
			s2:       []int{4, 3},
			expError: "malformed tensors",
		},

		// If the first argument is 2-dimensional and the second argument is 1-dimensional, the matrix-vector product is returned.
		{
			s1: []int{2, 4},
			s2: []int{4},
			expVals: []float32{
				14, 38,
			},
			expShape: []int{2},
		},
		{
			s1:       []int{2, 3},
			s2:       []int{4},
			expError: "malformed tensors",
		},
		{
			s1:       []int{4, 2},
			s2:       []int{4},
			expError: "malformed tensors",
		},

		// If both arguments are at least 1-dimensional and at least one
		// argument is N-dimensional (where N > 2), then a batched matrix
		// multiply is returned. If the first argument is 1-dimensional, a 1 is
		// prepended to its dimension for the purpose of the batched matrix
		// multiply and removed after. If the second argument is 1-dimensional,
		// a 1 is appended to its dimension for the purpose of the batched
		// matrix multiple and removed after. The non-matrix (i.e. batch)
		// dimensions are broadcasted (and thus must be broadcastable).
		{
			s1: []int{2, 4, 3},
			s2: []int{3},
			expVals: []float32{
				5, 14, 23, 32,
				41, 50, 59, 68,
			},
			expShape: []int{2, 4},
		},
		{
			s1: []int{2, 1, 3},
			s2: []int{3},
			expVals: []float32{
				5, 14,
			},
			expShape: []int{2, 1},
		},
		{
			s1:       []int{2, 4, 3},
			s2:       []int{4},
			expError: "malformed tensors",
		},

		{
			s1: []int{4},
			s2: []int{2, 4, 3},
			expVals: []float32{
				42, 48, 54,
				114, 120, 126,
			},
			expShape: []int{2, 3},
		},
		{
			s1: []int{4},
			s2: []int{2, 4, 1},
			expVals: []float32{
				14, 38,
			},
			expShape: []int{2, 1},
		},
		{
			s1: []int{4},
			s2: []int{1, 4, 3},
			expVals: []float32{
				42, 48, 54,
			},
			expShape: []int{1, 3},
		},
		{
			s1:       []int{3},
			s2:       []int{2, 4, 3},
			expError: "malformed tensors",
		},

		{
			s1: []int{4},
			s2: []int{2, 3, 4, 5},
			expVals: []float32{
				70, 76, 82, 88, 94,
				190, 196, 202, 208, 214,
				310, 316, 322, 328, 334,
				430, 436, 442, 448, 454,
				550, 556, 562, 568, 574,
				670, 676, 682, 688, 694,
			},
			expShape: []int{2, 3, 5},
		},
		{
			s1: []int{4},
			s2: []int{2, 1, 4, 5},
			expVals: []float32{
				70, 76, 82, 88, 94,
				190, 196, 202, 208, 214,
			},
			expShape: []int{2, 1, 5},
		},
		{
			s1: []int{4},
			s2: []int{2, 5, 4, 1},
			expVals: []float32{
				14, 38, 62, 86, 110,
				134, 158, 182, 206, 230,
			},
			expShape: []int{2, 5, 1},
		},
		{
			s1: []int{4},
			s2: []int{1, 5, 4, 2},
			expVals: []float32{
				28, 34,
				76, 82,
				124, 130,
				172, 178,
				220, 226,
			},
			expShape: []int{1, 5, 2},
		},

		{
			s1:       []int{2},
			s2:       []int{2, 3, 4, 5},
			expError: "malformed tensors",
		},
		{
			s1:       []int{3},
			s2:       []int{2, 3, 4, 5},
			expError: "malformed tensors",
		},
		{
			s1:       []int{5},
			s2:       []int{2, 3, 4, 5},
			expError: "malformed tensors",
		},
		{
			s1: []int{2, 3, 4, 5},
			s2: []int{5},
			expVals: []float32{
				30, 80, 130, 180,
				230, 280, 330, 380,
				430, 480, 530, 580,
				630, 680, 730, 780,
				830, 880, 930, 980,
				1030, 1080, 1130, 1180,
			},
			expShape: []int{2, 3, 4},
		},
		{
			s1:       []int{2, 3, 4, 5},
			s2:       []int{4},
			expError: "malformed tensors",
		},
		{
			s1:       []int{2, 3, 4, 5},
			s2:       []int{3},
			expError: "malformed tensors",
		},
		{
			s1:       []int{2, 3, 4, 5},
			s2:       []int{2},
			expError: "malformed tensors",
		},

		{
			s1: []int{4, 2},
			s2: []int{3, 2, 4},
			expVals: []float32{
				4, 5, 6, 7,
				12, 17, 22, 27,
				20, 29, 38, 47,
				28, 41, 54, 67,
				12, 13, 14, 15,
				52, 57, 62, 67,
				92, 101, 110, 119,
				132, 145, 158, 171,
				20, 21, 22, 23,
				92, 97, 102, 107,
				164, 173, 182, 191,
				236, 249, 262, 275,
			},
			expShape: []int{3, 4, 4},
		},
		{
			s1: []int{3, 2},
			s2: []int{3, 2, 4},
			expVals: []float32{
				4, 5, 6, 7,
				12, 17, 22, 27,
				20, 29, 38, 47,
				12, 13, 14, 15,
				52, 57, 62, 67,
				92, 101, 110, 119,
				20, 21, 22, 23,
				92, 97, 102, 107,
				164, 173, 182, 191,
			},
			expShape: []int{3, 3, 4},
		},
		{
			s1: []int{2, 3},
			s2: []int{2, 3, 4},
			expVals: []float32{
				20, 23, 26, 29,
				56, 68, 80, 92,
				56, 59, 62, 65,
				200, 212, 224, 236,
			},
			expShape: []int{2, 2, 4},
		},
		{
			s1: []int{2, 3},
			s2: []int{2, 3, 1},
			expVals: []float32{
				5, 14, 14, 50,
			},
			expShape: []int{2, 2, 1},
		},
		{
			s1: []int{2, 3},
			s2: []int{1, 3, 2},
			expVals: []float32{
				10, 13, 28, 40,
			},
			expShape: []int{1, 2, 2},
		},

		{
			s1:       []int{2, 4},
			s2:       []int{3, 2, 4},
			expError: "malformed tensors",
		},
		{
			s1:       []int{2, 3},
			s2:       []int{3, 2, 4},
			expError: "malformed tensors",
		},
		{
			s1: []int{2, 4},
			s2: []int{2, 4, 2},
			expVals: []float32{
				28, 34, 76, 98, 76, 82, 252, 274,
			},
			expShape: []int{2, 2, 2},
		},
		{
			s1: []int{2, 3, 4},
			s2: []int{4, 3},
			expVals: []float32{
				42, 48, 54,
				114, 136, 158,
				186, 224, 262,
				258, 312, 366,
				330, 400, 470,
				402, 488, 574,
			},
			expShape: []int{2, 3, 3},
		},
		{
			s1: []int{2, 3, 4},
			s2: []int{4, 6},
			expVals: []float32{
				84, 90, 96, 102, 108, 114,
				228, 250, 272, 294, 316, 338,
				372, 410, 448, 486, 524, 562,
				516, 570, 624, 678, 732, 786,
				660, 730, 800, 870, 940, 1010,
				804, 890, 976, 1062, 1148, 1234,
			},
			expShape: []int{2, 3, 6},
		},
		{
			s1: []int{2, 3, 4},
			s2: []int{4, 1},
			expVals: []float32{
				14, 38, 62, 86, 110, 134,
			},
			expShape: []int{2, 3, 1},
		},
		{
			s1: []int{2, 1, 4},
			s2: []int{4, 2},
			expVals: []float32{
				28, 34, 76, 98,
			},
			expShape: []int{2, 1, 2},
		},
		{
			s1: []int{1, 2, 4},
			s2: []int{4, 2},
			expVals: []float32{
				28, 34, 76, 98,
			},
			expShape: []int{1, 2, 2},
		},
		{
			s1:       []int{2, 3, 4},
			s2:       []int{3, 4},
			expError: "malformed tensors",
		},

		{
			s1: []int{3, 4},
			s2: []int{2, 2, 4, 3},
			expVals: []float32{
				42, 48, 54,
				114, 136, 158,
				186, 224, 262,
				114, 120, 126,
				378, 400, 422,
				642, 680, 718,
				186, 192, 198,
				642, 664, 686,
				1098, 1136, 1174,
				258, 264, 270,
				906, 928, 950,
				1554, 1592, 1630,
			},
			expShape: []int{2, 2, 3, 3},
		},
		{
			s1: []int{4, 5},
			s2: []int{2, 3, 5, 4},
			expVals: []float32{
				120, 130, 140, 150,
				320, 355, 390, 425,
				520, 580, 640, 700,
				720, 805, 890, 975,
				320, 330, 340, 350,
				1020, 1055, 1090, 1125,
				1720, 1780, 1840, 1900,
				2420, 2505, 2590, 2675,
				520, 530, 540, 550,
				1720, 1755, 1790, 1825,
				2920, 2980, 3040, 3100,
				4120, 4205, 4290, 4375,
				720, 730, 740, 750,
				2420, 2455, 2490, 2525,
				4120, 4180, 4240, 4300,
				5820, 5905, 5990, 6075,
				920, 930, 940, 950,
				3120, 3155, 3190, 3225,
				5320, 5380, 5440, 5500,
				7520, 7605, 7690, 7775,
				1120, 1130, 1140, 1150,
				3820, 3855, 3890, 3925,
				6520, 6580, 6640, 6700,
				9220, 9305, 9390, 9475,
			},
			expShape: []int{2, 3, 4, 4},
		},
		{
			s1: []int{2, 5},
			s2: []int{2, 3, 5, 4},
			expVals: []float32{
				120, 130, 140, 150,
				320, 355, 390, 425,
				320, 330, 340, 350,
				1020, 1055, 1090, 1125,
				520, 530, 540, 550,
				1720, 1755, 1790, 1825,
				720, 730, 740, 750,
				2420, 2455, 2490, 2525,
				920, 930, 940, 950,
				3120, 3155, 3190, 3225,
				1120, 1130, 1140, 1150,
				3820, 3855, 3890, 3925,
			},
			expShape: []int{2, 3, 2, 4},
		},
		{
			s1: []int{2, 5},
			s2: []int{2, 3, 5, 1},
			expVals: []float32{
				30, 80, 80, 255, 130, 430, 180, 605, 230, 780, 280, 955,
			},
			expShape: []int{2, 3, 2, 1},
		},
		{
			s1: []int{2, 5},
			s2: []int{2, 1, 5, 3},
			expVals: []float32{
				90, 100, 110,
				240, 275, 310,
				240, 250, 260,
				765, 800, 835,
			},
			expShape: []int{2, 1, 2, 3},
		},
		{
			s1: []int{2, 5},
			s2: []int{1, 2, 5, 3},
			expVals: []float32{
				90, 100, 110,
				240, 275, 310,
				240, 250, 260,
				765, 800, 835,
			},
			expShape: []int{1, 2, 2, 3},
		},
		{
			s1: []int{1, 5},
			s2: []int{1, 2, 5, 3},
			expVals: []float32{
				90, 100, 110,
				240, 250, 260,
			},
			expShape: []int{1, 2, 1, 3},
		},

		{
			s1:       []int{2, 4},
			s2:       []int{2, 3, 5, 4},
			expError: "malformed tensors",
		},
		{
			s1:       []int{5, 2},
			s2:       []int{2, 3, 5, 4},
			expError: "malformed tensors",
		},

		{
			s1: []int{5, 3, 2, 4},
			s2: []int{4, 2},
			expVals: []float32{
				28, 34, 76, 98,
				124, 162, 172, 226,
				220, 290, 268, 354,
				316, 418, 364, 482,
				412, 546, 460, 610,
				508, 674, 556, 738,
				604, 802, 652, 866,
				700, 930, 748, 994,
				796, 1058, 844, 1122,
				892, 1186, 940, 1250,
				988, 1314, 1036, 1378,
				1084, 1442, 1132, 1506,
				1180, 1570, 1228, 1634,
				1276, 1698, 1324, 1762,
				1372, 1826, 1420, 1890,
			},
			expShape: []int{5, 3, 2, 2},
		},
		{
			s1: []int{5, 3, 2, 4},
			s2: []int{4, 3},
			expVals: []float32{
				42, 48, 54, 114, 136, 158,
				186, 224, 262, 258, 312, 366,
				330, 400, 470, 402, 488, 574,
				474, 576, 678, 546, 664, 782,
				618, 752, 886, 690, 840, 990,
				762, 928, 1094, 834, 1016, 1198,
				906, 1104, 1302, 978, 1192, 1406,
				1050, 1280, 1510, 1122, 1368, 1614,
				1194, 1456, 1718, 1266, 1544, 1822,
				1338, 1632, 1926, 1410, 1720, 2030,
				1482, 1808, 2134, 1554, 1896, 2238,
				1626, 1984, 2342, 1698, 2072, 2446,
				1770, 2160, 2550, 1842, 2248, 2654,
				1914, 2336, 2758, 1986, 2424, 2862,
				2058, 2512, 2966, 2130, 2600, 3070,
			},
			expShape: []int{5, 3, 2, 3},
		},
		{
			s1: []int{5, 3, 2, 4},
			s2: []int{4, 1},
			expVals: []float32{
				14, 38, 62, 86, 110, 134, 158, 182, 206, 230, 254, 278, 302, 326, 350, 374, 398, 422, 446, 470, 494, 518, 542, 566, 590, 614, 638, 662, 686, 710,
			},
			expShape: []int{5, 3, 2, 1},
		},
		{
			s1: []int{5, 2, 1, 4},
			s2: []int{4, 1},
			expVals: []float32{
				14, 38, 62, 86, 110, 134, 158, 182, 206, 230,
			},
			expShape: []int{5, 2, 1, 1},
		},
		{
			s1: []int{5, 1, 2, 4},
			s2: []int{4, 2},
			expVals: []float32{
				28, 34, 76, 98,
				124, 162, 172, 226,
				220, 290, 268, 354,
				316, 418, 364, 482,
				412, 546, 460, 610,
			},
			expShape: []int{5, 1, 2, 2},
		},
		{
			s1: []int{1, 3, 2, 4},
			s2: []int{4, 2},
			expVals: []float32{
				28, 34, 76, 98,
				124, 162, 172, 226,
				220, 290, 268, 354,
			},
			expShape: []int{1, 3, 2, 2},
		},

		{
			s1:       []int{5, 3, 2, 4},
			s2:       []int{2, 4},
			expError: "malformed tensors",
		},
		{
			s1:       []int{5, 3, 2, 4},
			s2:       []int{2, 3},
			expError: "malformed tensors",
		},
		{
			s1:       []int{5, 3, 2, 4},
			s2:       []int{3, 2},
			expError: "malformed tensors",
		},

		{
			s1: []int{2, 3, 4},
			s2: []int{5, 2, 4, 3},
			expVals: []float32{
				42, 48, 54, 114, 136, 158, 186, 224, 262,
				906, 960, 1014, 1170, 1240, 1310, 1434, 1520, 1606,
				186, 192, 198, 642, 664, 686, 1098, 1136, 1174,
				2202, 2256, 2310, 2850, 2920, 2990, 3498, 3584, 3670,
				330, 336, 342, 1170, 1192, 1214, 2010, 2048, 2086,
				3498, 3552, 3606, 4530, 4600, 4670, 5562, 5648, 5734,
				474, 480, 486, 1698, 1720, 1742, 2922, 2960, 2998,
				4794, 4848, 4902, 6210, 6280, 6350, 7626, 7712, 7798,
				618, 624, 630, 2226, 2248, 2270, 3834, 3872, 3910,
				6090, 6144, 6198, 7890, 7960, 8030, 9690, 9776, 9862,
			},
			expShape: []int{5, 2, 3, 3},
		},
		{
			s1: []int{1, 4, 3},
			s2: []int{2, 2, 3, 4},
			expVals: []float32{
				20, 23, 26, 29,
				56, 68, 80, 92,
				92, 113, 134, 155,
				128, 158, 188, 218,
				56, 59, 62, 65,
				200, 212, 224, 236,
				344, 365, 386, 407,
				488, 518, 548, 578,
				92, 95, 98, 101,
				344, 356, 368, 380,
				596, 617, 638, 659,
				848, 878, 908, 938,
				128, 131, 134, 137,
				488, 500, 512, 524,
				848, 869, 890, 911,
				1208, 1238, 1268, 1298,
			},
			expShape: []int{2, 2, 4, 4},
		},

		{
			s1:       []int{2, 4, 3},
			s2:       []int{5, 2, 4, 3},
			expError: "malformed tensors",
		},

		{
			s1: []int{5, 2, 3, 4},
			s2: []int{2, 4, 3},
			expVals: []float32{
				42, 48, 54, 114, 136, 158, 186, 224, 262,
				906, 960, 1014, 1170, 1240, 1310, 1434, 1520, 1606,
				474, 576, 678, 546, 664, 782, 618, 752, 886,
				2490, 2640, 2790, 2754, 2920, 3086, 3018, 3200, 3382,
				906, 1104, 1302, 978, 1192, 1406, 1050, 1280, 1510,
				4074, 4320, 4566, 4338, 4600, 4862, 4602, 4880, 5158,
				1338, 1632, 1926, 1410, 1720, 2030, 1482, 1808, 2134,
				5658, 6000, 6342, 5922, 6280, 6638, 6186, 6560, 6934,
				1770, 2160, 2550, 1842, 2248, 2654, 1914, 2336, 2758,
				7242, 7680, 8118, 7506, 7960, 8414, 7770, 8240, 8710,
			},
			expShape: []int{5, 2, 3, 3},
		},
		{
			s1: []int{5, 2, 3, 4},
			s2: []int{2, 4, 2},
			expVals: []float32{
				28, 34, 76, 98, 124, 162,
				604, 658, 780, 850, 956, 1042,
				316, 418, 364, 482, 412, 546,
				1660, 1810, 1836, 2002, 2012, 2194,
				604, 802, 652, 866, 700, 930,
				2716, 2962, 2892, 3154, 3068, 3346,
				892, 1186, 940, 1250, 988, 1314,
				3772, 4114, 3948, 4306, 4124, 4498,
				1180, 1570, 1228, 1634, 1276, 1698,
				4828, 5266, 5004, 5458, 5180, 5650,
			},
			expShape: []int{5, 2, 3, 2},
		},

		{
			s1:       []int{5, 2, 3, 4},
			s2:       []int{2, 2, 3},
			expError: "malformed tensors",
		},
		{
			s1:       []int{5, 2, 3, 4},
			s2:       []int{5, 4, 3},
			expError: "cannot broadcast",
		},
		{
			s1:       []int{5, 2, 3, 4},
			s2:       []int{3, 4, 2},
			expError: "cannot broadcast",
		},

		{
			s1: []int{2, 3, 4},
			s2: []int{2, 4, 3},
			expVals: []float32{
				42, 48, 54, 114, 136, 158, 186, 224, 262,
				906, 960, 1014, 1170, 1240, 1310, 1434, 1520, 1606,
			},
			expShape: []int{2, 3, 3},
		},
		{
			s1:       []int{2, 3, 4},
			s2:       []int{2, 3, 4},
			expError: "malformed tensors",
		},
		{
			s1:       []int{2, 3, 4},
			s2:       []int{3, 4, 3},
			expError: "cannot broadcast",
		},
		{
			s1:       []int{2, 3, 4},
			s2:       []int{4, 4, 3},
			expError: "cannot broadcast",
		},
		// pytorch rejects [4,4,3] @ [2,3,4] but GGML accepts

		{
			s1: []int{2, 2, 3, 4},
			s2: []int{2, 2, 4, 3},
			expVals: []float32{
				42, 48, 54, 114, 136, 158, 186, 224, 262,
				906, 960, 1014, 1170, 1240, 1310, 1434, 1520, 1606,
				2922, 3024, 3126, 3378, 3496, 3614, 3834, 3968, 4102,
				6090, 6240, 6390, 6738, 6904, 7070, 7386, 7568, 7750,
			},
			expShape: []int{2, 2, 3, 3},
		},
		{
			s1: []int{2, 2, 3, 4},
			s2: []int{2, 2, 4, 2},
			expVals: []float32{
				28, 34, 76, 98, 124, 162,
				604, 658, 780, 850, 956, 1042,
				1948, 2050, 2252, 2370, 2556, 2690,
				4060, 4210, 4492, 4658, 4924, 5106,
			},
			expShape: []int{2, 2, 3, 2},
		},
		{
			s1: []int{2, 2, 3, 4},
			s2: []int{2, 1, 4, 2},
			expVals: []float32{
				28, 34, 76, 98, 124, 162,
				172, 226, 220, 290, 268, 354,
				1132, 1234, 1308, 1426, 1484, 1618,
				1660, 1810, 1836, 2002, 2012, 2194,
			},
			expShape: []int{2, 2, 3, 2},
		},
		{
			s1: []int{5, 1, 4, 3},
			s2: []int{5, 2, 3, 4},
			expVals: []float32{
				20, 23, 26, 29,
				56, 68, 80, 92,
				92, 113, 134, 155,
				128, 158, 188, 218,
				56, 59, 62, 65,
				200, 212, 224, 236,
				344, 365, 386, 407,
				488, 518, 548, 578,
				1100, 1139, 1178, 1217,
				1352, 1400, 1448, 1496,
				1604, 1661, 1718, 1775,
				1856, 1922, 1988, 2054,
				1568, 1607, 1646, 1685,
				1928, 1976, 2024, 2072,
				2288, 2345, 2402, 2459,
				2648, 2714, 2780, 2846,
				3908, 3983, 4058, 4133,
				4376, 4460, 4544, 4628,
				4844, 4937, 5030, 5123,
				5312, 5414, 5516, 5618,
				4808, 4883, 4958, 5033,
				5384, 5468, 5552, 5636,
				5960, 6053, 6146, 6239,
				6536, 6638, 6740, 6842,
				8444, 8555, 8666, 8777,
				9128, 9248, 9368, 9488,
				9812, 9941, 10070, 10199,
				10496, 10634, 10772, 10910,
				9776, 9887, 9998, 10109,
				10568, 10688, 10808, 10928,
				11360, 11489, 11618, 11747,
				12152, 12290, 12428, 12566,
				14708, 14855, 15002, 15149,
				15608, 15764, 15920, 16076,
				16508, 16673, 16838, 17003,
				17408, 17582, 17756, 17930,
				16472, 16619, 16766, 16913,
				17480, 17636, 17792, 17948,
				18488, 18653, 18818, 18983,
				19496, 19670, 19844, 20018,
			},
			expShape: []int{5, 2, 4, 4},
		},

		{
			s1: []int{2, 3, 4, 5},
			s2: []int{2, 3, 5, 4},
			expVals: []float32{
				120, 130, 140, 150,
				320, 355, 390, 425,
				520, 580, 640, 700,
				720, 805, 890, 975,
				3120, 3230, 3340, 3450,
				3820, 3955, 4090, 4225,
				4520, 4680, 4840, 5000,
				5220, 5405, 5590, 5775,
				10120, 10330, 10540, 10750,
				11320, 11555, 11790, 12025,
				12520, 12780, 13040, 13300,
				13720, 14005, 14290, 14575,
				21120, 21430, 21740, 22050,
				22820, 23155, 23490, 23825,
				24520, 24880, 25240, 25600,
				26220, 26605, 26990, 27375,
				36120, 36530, 36940, 37350,
				38320, 38755, 39190, 39625,
				40520, 40980, 41440, 41900,
				42720, 43205, 43690, 44175,
				55120, 55630, 56140, 56650,
				57820, 58355, 58890, 59425,
				60520, 61080, 61640, 62200,
				63220, 63805, 64390, 64975,
			},
			expShape: []int{2, 3, 4, 4},
		},
	}

	be := newTestBackend(100)
	ctx := newTestContext(be, 100).Input()

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("s1:%s_s2:%s",
			strings.Trim(strings.Join(strings.Fields(fmt.Sprint(tc.s1)), ","), "[]"),
			strings.Trim(strings.Join(strings.Fields(fmt.Sprint(tc.s2)), ","), "[]"),
		), func(t *testing.T) {

			t1 := arange(t, ctx, tc.s1)
			slog.Info("Created", "t1", t1)
			ctx.Forward(t1)
			fmt.Fprintln(os.Stderr, ml.Dump(ctx, t1))
			t2 := arange(t, ctx, tc.s2)
			slog.Info("Created", "t2", t2)
			ctx.Forward(t2)
			fmt.Fprintln(os.Stderr, ml.Dump(ctx, t2))
			if tc.expError != "" {
				defer func() {
					e := recover()
					if !strings.Contains(fmt.Sprintf("%v", e), tc.expError) {
						t.Fatalf("incorrect panic\ngot %v\nexpected %s", e, tc.expError)
					}
				}()
			} else {
				defer func() {
					e := recover()
					if e != nil {
						t.Fatalf("hit unexpected panic: %v", e)
					}
				}()

			}
			t3 := t1.Matmul(ctx, t2) //.Contiguous(ctx)
			slog.Info("Result", "t3", t3)
			ctx.Forward(t3)
			if tc.expError != "" {
				t.Fatalf("expected error %s but Matmul did not panic", tc.expError)
			}
			fmt.Fprintf(os.Stderr, "Dumping t1 @ t2 result\n")
			fmt.Fprintln(os.Stderr, ml.Dump(ctx, t3))
			res := t3.Floats()
			if !reflect.DeepEqual(res, tc.expVals) || !reflect.DeepEqual(t3.Shape(), tc.expShape) {
				t.Fatalf("incorrect result\ngot %v %v\nexpected %v %v", t3.Shape(), res, tc.expShape, tc.expVals)
			}
		})
	}
}
