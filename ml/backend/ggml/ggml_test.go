package ggml

import (
	"fmt"
	"log/slog"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/ml"
)

func init() {
	if envconfig.Debug() {
		logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelDebug}))
		slog.SetDefault(logger)
	}
}

func dump(msg string) {
	if envconfig.Debug() {
		fmt.Fprintln(os.Stderr, msg)
	}
}

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
	slog.Debug("Initial data", "tensor", x)
	for _, tc := range permutes[2] {
		t.Run(fmt.Sprintf("%v", tc), func(t *testing.T) {
			t2 := x.Permute(ctx, tc...)
			slog.Debug("After permute", "request", tc, "result", t2)
			res := t2.Shape()
			expected := [2]int{
				shape[tc[0]],
				shape[tc[1]],
			}
			if ([2]int)(res) != expected {
				t.Fatalf("reshape %v expected %v but got %v", tc, expected, res)
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
	slog.Debug("Initial data", "tensor", x)
	for _, tc := range permutes[3] {
		t.Run(fmt.Sprintf("%v", tc), func(t *testing.T) {
			t2 := x.Permute(ctx, tc...)
			slog.Debug("After permute", "request", tc, "result", t2)
			res := t2.Shape()
			expected := [3]int{
				shape[tc[0]],
				shape[tc[1]],
				shape[tc[2]],
			}
			if ([3]int)(res) != expected {
				t.Fatalf("reshape %v expected %v but got %v", tc, expected, res)
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
	slog.Debug("Initial data", "tensor", x)
	for _, tc := range permutes[4] {
		t.Run(fmt.Sprintf("%v", tc), func(t *testing.T) {
			t2 := x.Permute(ctx, tc...)
			slog.Debug("After permute", "request", tc, "result", t2)
			res := t2.Shape()
			expected := [4]int{
				shape[tc[0]],
				shape[tc[1]],
				shape[tc[2]],
				shape[tc[3]],
			}
			if ([4]int)(res) != expected {
				t.Fatalf("reshape %v expected %v but got %v", tc, expected, res)
			}
		})
	}
}

func TestMul(t *testing.T) {
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
			slog.Debug("Created", "t1", t1)
			ctx.Forward(t1)
			dump(ml.Dump(ctx, t1))
			t2 := arange(t, ctx, tc.s2)
			slog.Debug("Created", "t2", t2)
			ctx.Forward(t2)
			dump(ml.Dump(ctx, t2))
			t3 := t1.Mul(ctx, t2)
			ctx.Forward(t3)
			dump("Dumping t1 * t2 result\n")
			dump(ml.Dump(ctx, t3))
			res := t3.Floats()
			if !reflect.DeepEqual(res, tc.expVals) {
				t.Fatalf("incorrect result\ngot %v\nexpected %v", res, tc.expVals)
			}
		})
	}
}

func TestMatmul(t *testing.T) {
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
			slog.Debug("Created", "t1", t1)
			ctx.Forward(t1)
			dump(ml.Dump(ctx, t1))
			t2 := arange(t, ctx, tc.s2)
			slog.Debug("Created", "t2", t2)
			ctx.Forward(t2)
			dump(ml.Dump(ctx, t2))
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
			slog.Debug("Result", "t3", t3)
			ctx.Forward(t3)
			if tc.expError != "" {
				t.Fatalf("expected error %s but Matmul did not panic", tc.expError)
			}
			dump("Dumping t1 @ t2 result\n")
			dump(ml.Dump(ctx, t3))
			res := t3.Floats()
			if !reflect.DeepEqual(res, tc.expVals) || !reflect.DeepEqual(t3.Shape(), tc.expShape) {
				t.Fatalf("incorrect result\ngot %v %v\nexpected %v %v", t3.Shape(), res, tc.expShape, tc.expVals)
			}
		})
	}
}

func TestAsStrided(t *testing.T) {
	/* Equivalent to the following pytorch - all expValues and expShapes derived from this

	import torch
	t1 = torch.arange(12).view(3,4)
	print(t1)
	t2 = torch.as_strided(t1, (2,2), (1,2), 1)
	print(t2)
	*/

	type testCase struct {
		sh1      []int
		sh2      []int
		st2      []int
		offset   int
		expVals  []float32
		expError string
	}
	testCases := []testCase{
		{
			sh1:    []int{4, 5},
			sh2:    []int{2, 5},
			st2:    []int{5, 1},
			offset: 5,
			expVals: []float32{
				5, 6, 7, 8, 9,
				10, 11, 12, 13, 14,
			},
		},
		{
			sh1:    []int{4, 4},
			sh2:    []int{2, 4},
			st2:    []int{4, 1},
			offset: 0,
			expVals: []float32{
				0, 1, 2, 3,
				4, 5, 6, 7,
			},
		},
		{
			sh1:    []int{2, 3, 4},
			sh2:    []int{1, 3, 4},
			st2:    []int{1, 4, 1},
			offset: 12,
			expVals: []float32{
				12, 13, 14, 15,
				16, 17, 18, 19,
				20, 21, 22, 23,
			},
		},
		{
			sh1:    []int{2, 3, 4},
			sh2:    []int{3, 4},
			st2:    []int{4, 1},
			offset: 12,
			expVals: []float32{
				12, 13, 14, 15,
				16, 17, 18, 19,
				20, 21, 22, 23,
			},
		},
		{
			sh1:    []int{2, 3, 4},
			sh2:    []int{2, 4},
			st2:    []int{4, 1},
			offset: 8,
			expVals: []float32{
				8, 9, 10, 11,
				12, 13, 14, 15,
			},
		},
		{
			sh1:    []int{2, 3, 4},
			sh2:    []int{4},
			st2:    []int{1},
			offset: 8,
			expVals: []float32{
				8, 9, 10, 11,
			},
		},
		{
			sh1:    []int{2, 3, 4},
			sh2:    []int{3, 2},
			st2:    []int{2, 1},
			offset: 8,
			expVals: []float32{
				8, 9, 10, 11, 12, 13,
			},
		},
		{
			sh1:    []int{2, 3, 4},
			sh2:    []int{3, 2},
			st2:    []int{3, 1},
			offset: 8,
			expVals: []float32{
				8, 9, 11, 12, 14, 15,
			},
		},
		{
			sh1:    []int{2, 3, 4},
			sh2:    []int{3, 2},
			st2:    []int{4, 1},
			offset: 8,
			expVals: []float32{
				8, 9, 12, 13, 16, 17,
			},
		},
		{
			sh1:    []int{2, 3, 4},
			sh2:    []int{3, 2},
			st2:    []int{1, 1},
			offset: 8,
			expVals: []float32{
				8, 9, 9, 10, 10, 11,
			},
		},
		{
			sh1:    []int{2, 3, 4},
			sh2:    []int{3, 2},
			st2:    []int{0, 1},
			offset: 8,
			expVals: []float32{
				8, 9, 8, 9, 8, 9,
			},
		},
		{
			sh1:    []int{3, 2, 4},
			sh2:    []int{1, 2, 4},
			st2:    []int{1, 4, 1},
			offset: 8,
			expVals: []float32{
				8, 9, 10, 11,
				12, 13, 14, 15,
			},
		},

		{
			sh1:    []int{3, 2, 4},
			sh2:    []int{1, 2, 4},
			st2:    []int{99, 4, 1}, // Note: pytorch reports the view with 99 stride, ggml does not
			offset: 8,
			expVals: []float32{
				8, 9, 10, 11,
				12, 13, 14, 15,
			},
		},
		{
			sh1:    []int{4, 2, 4},
			sh2:    []int{2, 2, 4},
			st2:    []int{8, 4, 1},
			offset: 8,
			expVals: []float32{
				8, 9, 10, 11,
				12, 13, 14, 15,

				16, 17, 18, 19,
				20, 21, 22, 23,
			},
		},
		{
			sh1:    []int{4, 2, 4},
			sh2:    []int{2, 2, 4},
			st2:    []int{4, 4, 1},
			offset: 8,
			expVals: []float32{
				8, 9, 10, 11,
				12, 13, 14, 15,

				12, 13, 14, 15,
				16, 17, 18, 19,
			},
		},
		{
			sh1:    []int{4, 2, 4},
			sh2:    []int{2, 2, 4},
			st2:    []int{16, 4, 1},
			offset: 8,
			expVals: []float32{
				8, 9, 10, 11,
				12, 13, 14, 15,

				24, 25, 26, 27,
				28, 29, 30, 31,
			},
		},

		{
			sh1: []int{4, 2, 32},
			sh2: []int{2, 16},
			st2: []int{32, 1},
			expVals: []float32{
				96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
				110, 111,
				128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
				142, 143,
			},
			offset: 96,
		},

		{ // 1/2 row, skip a row
			sh1: []int{4, 2, 32},
			sh2: []int{2, 8},
			st2: []int{16, 1},
			expVals: []float32{
				96, 97, 98, 99, 100, 101, 102, 103,
				112, 113, 114, 115, 116, 117, 118, 119,
			},
			offset: 96,
		},

		// Error cases (both pytorch and ggml)
		{
			sh1:      []int{2, 3, 4},
			sh2:      []int{3, 2},
			st2:      []int{1},
			offset:   8,
			expError: "mismatch in length",
		},
		{
			sh1:      []int{2, 3, 4},
			sh2:      []int{3},
			st2:      []int{1, 1},
			offset:   8,
			expError: "mismatch in length",
		},
		// Limitations of GGML (supported by pytorch)
		{
			sh1:    []int{2, 3, 4},
			sh2:    []int{3, 2},
			st2:    []int{2, 2},
			offset: 8,
			expVals: []float32{
				8, 10, 10, 12, 12, 14, // pytorch output
			},
			expError: "final stride must be 1",
		},
	}

	be := newTestBackend(100)
	ctx := newTestContext(be, 100).Input()

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("sh1:%s_sh2:%s_st2:%s_offset:%d",
			strings.Trim(strings.Join(strings.Fields(fmt.Sprint(tc.sh1)), ","), "[]"),
			strings.Trim(strings.Join(strings.Fields(fmt.Sprint(tc.sh2)), ","), "[]"),
			strings.Trim(strings.Join(strings.Fields(fmt.Sprint(tc.st2)), ","), "[]"),
			tc.offset,
		), func(t *testing.T) {

			t1 := arange(t, ctx, tc.sh1)
			slog.Debug("Created", "t1", t1)
			ctx.Forward(t1)
			dump(ml.Dump(ctx, t1))
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
			t2 := t1.AsStrided(ctx, tc.sh2, tc.st2, tc.offset).Contiguous(ctx)
			slog.Debug("Result", "t2", t2)
			ctx.Forward(t2)
			if tc.expError != "" {
				t.Fatalf("expected error %s but AsStrided did not panic", tc.expError)
			}
			dump(fmt.Sprintf("Dumping t2 result shape:%v\n", t2.Shape()))
			dump(ml.Dump(ctx, t2))
			res := t2.Floats()
			if !reflect.DeepEqual(res, tc.expVals) || !reflect.DeepEqual(t2.Shape(), tc.sh2) {
				t.Fatalf("incorrect result\ngot %v %v\nexpected %v %v", t2.Shape(), res, tc.sh2, tc.expVals)
			}
		})
	}
}

func TestPermutedAsStrided(t *testing.T) {
	/* Equivalent to the following pytorch - all expected results derived from this

	import torch
	t1 = torch.arange(12).view(3,4).permuted((1,0))
	print(t1)
	t2 = torch.as_strided(t1, (2,2), (1,2), 1)
	print(t2)
	*/

	type testCase struct {
		sh1      []int // Initial shape
		p1       []int // Permute call on initial tensor
		sh2      []int // view shape
		st2      []int // view stride
		offset   int   // view offset
		expVals  []float32
		expError string
	}
	testCases := []testCase{
		{
			sh1:    []int{2, 3, 4},
			p1:     []int{0, 2, 1, 3},
			sh2:    []int{1, 4, 3},
			st2:    []int{1, 3, 1},
			offset: 12,
			expVals: []float32{
				12, 13, 14,
				15, 16, 17,
				18, 19, 20,
				21, 22, 23,
			},
		},
		{
			sh1:    []int{2, 3, 4},
			p1:     []int{0, 2, 1, 3},
			sh2:    []int{1, 4, 3},
			st2:    []int{1, 2, 1},
			offset: 12,
			expVals: []float32{
				12, 13, 14,
				14, 15, 16,
				16, 17, 18,
				18, 19, 20,
			},
		},
		{
			sh1:    []int{2, 3, 4},
			p1:     []int{0, 2, 1, 3},
			sh2:    []int{1, 3, 4},
			st2:    []int{1, 2, 1},
			offset: 12,
			expVals: []float32{
				12, 13, 14, 15,
				14, 15, 16, 17,
				16, 17, 18, 19,
			},
		},
		{
			sh1:    []int{2, 3, 4},
			p1:     []int{1, 0, 2, 3},
			sh2:    []int{1, 2, 4},
			st2:    []int{1, 2, 1},
			offset: 8,
			expVals: []float32{
				8, 9, 10, 11,
				10, 11, 12, 13,
			},
		},
	}

	be := newTestBackend(100)
	ctx := newTestContext(be, 100).Input()

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("sh1:%s_sh2:p1:%s_%s_st2:%s_offset:%d",
			strings.Trim(strings.Join(strings.Fields(fmt.Sprint(tc.sh1)), ","), "[]"),
			strings.Trim(strings.Join(strings.Fields(fmt.Sprint(tc.p1)), ","), "[]"),
			strings.Trim(strings.Join(strings.Fields(fmt.Sprint(tc.sh2)), ","), "[]"),
			strings.Trim(strings.Join(strings.Fields(fmt.Sprint(tc.st2)), ","), "[]"),
			tc.offset,
		), func(t *testing.T) {

			t1 := arange(t, ctx, tc.sh1)
			slog.Debug("Created", "t1", t1)
			ctx.Forward(t1)
			dump(ml.Dump(ctx, t1))
			t2 := t1.Permute(ctx, tc.p1...)
			slog.Debug("Permuted", "t2", t2)
			ctx.Forward(t2)
			dump(ml.Dump(ctx, t2))
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
			t3 := t2.AsStrided(ctx, tc.sh2, tc.st2, tc.offset).Contiguous(ctx)
			slog.Debug("AsStrided", "t3", t3)
			ctx.Forward(t2) //.Compute(t2)
			if tc.expError != "" {
				t.Fatalf("expected error %s but AsStrided did not panic", tc.expError)
			}
			dump(fmt.Sprintf("Dumping t3 result shape:%v\n", t3.Shape()))
			dump(ml.Dump(ctx, t3))
			res := t3.Floats()
			if !reflect.DeepEqual(res, tc.expVals) || !reflect.DeepEqual(t3.Shape(), tc.sh2) {
				t.Fatalf("incorrect result\ngot %v %v\nexpected %v %v", t3.Shape(), res, tc.sh2, tc.expVals)
			}
		})
	}
}

func TestStride(t *testing.T) {
	/* Equivalent to the following pytorch - all expValues and expShapes derived from this

	import torch
	t1 = torch.arange(12).view(3,4)
	print(t1)
	print(t1.stride())
	*/
	type testCase struct {
		sh1       []int
		dtype     ml.DType
		expStride []int
	}
	testCases := []testCase{
		{
			sh1:       []int{2, 3, 4, 5},
			dtype:     ml.DTypeF32,
			expStride: []int{60, 20, 5, 1},
		},
		{
			sh1:       []int{2, 3, 4, 5},
			dtype:     ml.DTypeF16,
			expStride: []int{60, 20, 5, 1},
		},
		// TODO needs coverage....
		// {
		// 	sh1:      []int{2, 3, 4, 5},
		// 	dtype:    ml.DTypeI32,
		// 	expShape: []int{60, 20, 5, 1},
		// },
	}

	be := newTestBackend(100)
	ctx := newTestContext(be, 100).Input()

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("sh1:%s_dt:%s",
			strings.Trim(strings.Join(strings.Fields(fmt.Sprint(tc.sh1)), ","), "[]"),
			tc.dtype,
		), func(t *testing.T) {

			t1 := arange(t, ctx, tc.sh1)
			slog.Debug("Created", "t1", t1)
			defer func() {
				e := recover()
				if e != nil {
					t.Fatalf("hit unexpected panic: %v", e)
				}
			}()
			ctx.Forward(t1)
			dump(ml.Dump(ctx, t1))

			st := make([]int, len(tc.sh1))
			for i := range st {
				st[i] = t1.Stride(i)
			}
			if !reflect.DeepEqual(st, tc.expStride) {
				t.Fatalf("incorrect result\ngot %v\nexpected %v", st, tc.expStride)
			}
		})
	}
}

func TestPermutedStride(t *testing.T) {
	/* Equivalent to the following pytorch - all expValues and expShapes derived from this

	import torch
	t1 = torch.arange(12).view(3,4).permuted((1,0))
	print(t1)
	print(t1.stride())
	*/
	type testCase struct {
		sh1   []int // Tensor shape
		st1   []int // stride for initial tensor, used to derive expected strides
		dtype ml.DType
	}
	testCases := []testCase{
		{
			sh1:   []int{5, 7},
			st1:   []int{7, 1},
			dtype: ml.DTypeF16,
		},
		{
			sh1:   []int{3, 4, 5},
			st1:   []int{20, 5, 1},
			dtype: ml.DTypeF16,
		},
		{
			sh1:   []int{2, 3, 4, 5},
			st1:   []int{60, 20, 5, 1},
			dtype: ml.DTypeF32,
		},
		{
			sh1:   []int{2, 3, 4, 5},
			st1:   []int{60, 20, 5, 1},
			dtype: ml.DTypeF16,
		},
	}

	be := newTestBackend(100)
	ctx := newTestContext(be, 100).Input()

	for _, tc := range testCases {
		for _, p1 := range permutes[len(tc.sh1)] {
			t.Run(fmt.Sprintf("permute:%s/sh1:%s_dt:%s",
				strings.Trim(strings.Join(strings.Fields(fmt.Sprint(p1)), ","), "[]"),
				strings.Trim(strings.Join(strings.Fields(fmt.Sprint(tc.sh1)), ","), "[]"),
				tc.dtype,
			), func(t *testing.T) {

				t1 := arange(t, ctx, tc.sh1)
				slog.Debug("Created", "t1", t1)
				defer func() {
					e := recover()
					if e != nil {
						t.Fatalf("hit unexpected panic: %v", e)
					}
				}()
				ctx.Forward(t1)
				dump(ml.Dump(ctx, t1))
				t2 := t1.Permute(ctx, p1...)
				slog.Debug("Permuted", "t2", t2)
				ctx.Forward(t2)
				dump(ml.Dump(ctx, t2))

				st := make([]int, len(tc.sh1))
				for i := range st {
					st[i] = t2.Stride(i)
				}
				expStride := expStride(tc.st1, p1)
				if !reflect.DeepEqual(st, expStride) {
					t.Fatalf("incorrect result\ngot %v\nexpected %v", st, expStride)
				}
			})
		}
	}
}

func TestQuantStride(t *testing.T) {
	type testCase struct {
		sh1       []int
		expStride []int
	}
	testCases := []testCase{
		{
			sh1:       []int{8, 32},
			expStride: []int{32, 1},
		},
		{
			sh1:       []int{2, 4, 32},
			expStride: []int{128, 32, 1},
		},
		{
			sh1:       []int{2, 2, 2, 32},
			expStride: []int{128, 64, 32, 1},
		},
		{
			sh1:       []int{4, 2, 32},
			expStride: []int{64, 32, 1},
		},
		{
			sh1:       []int{2, 4, 2, 16},
			expStride: []int{128, 32, 16, 1},
		},
	}

	be := newTestBackend(100)
	ctx := newTestContext(be, 100).Input()

	for _, dtype := range []ml.DType{ml.DTypeQ40, ml.DTypeQ80} {
		for _, tc := range testCases {
			t.Run(fmt.Sprintf("%s/sh1:%s",
				dtype,
				strings.Trim(strings.Join(strings.Fields(fmt.Sprint(tc.sh1)), ","), "[]"),
			), func(t *testing.T) {
				t1 := ctx.(*Context).FromBytes(dtype, quantData[dtype], tc.sh1...)
				slog.Debug("Created", "t1", t1)
				defer func() {
					e := recover()
					if e != nil {
						t.Fatalf("hit unexpected panic: %v", e)
					}
				}()
				ctx.Forward(t1)
				dump(ml.Dump(ctx, t1))

				sh := make([]int, len(tc.sh1))
				for i := range sh {
					sh[i] = t1.Stride(i)
				}
				if !reflect.DeepEqual(sh, tc.expStride) {
					t.Fatalf("incorrect result\ngot %v\nexpected %v", sh, tc.expStride)
				}
			})
		}
	}
}

func TestPermutedQuantStride(t *testing.T) {
	/* Equivalent to the following pytorch - all expValues and expShapes derived from this

	import torch
	import gguf
	quant = gguf.GGMLQuantizationType.Q4_0
	shape = (2,4,32)
	t1 = np.arange(math.prod(shape), dtype=np.float16).reshape(shape)
	t2 = gguf.quantize(t1, quant)
	*/
	type testCase struct {
		sh1 []int // Must be large enough to fit block size and type size
		st1 []int // stride for initial tensor, used to derive expected strides
	}
	testCases := []testCase{
		{
			sh1: []int{8, 32},
			st1: []int{32, 1},
		},
		{
			sh1: []int{2, 4, 32},
			st1: []int{128, 32, 1},
		},
		{
			sh1: []int{2, 1, 2, 32},
			st1: []int{64, 64, 32, 1},
		},
	}

	be := newTestBackend(100)
	ctx := newTestContext(be, 100).Input()

	for _, dtype := range []ml.DType{ml.DTypeQ40, ml.DTypeQ80} {
		for _, tc := range testCases {
			for _, p1 := range permutes[len(tc.sh1)] {
				t.Run(fmt.Sprintf("%s/permuted:%s/sh1:%s",
					dtype,
					strings.Trim(strings.Join(strings.Fields(fmt.Sprint(p1)), ","), "[]"),
					strings.Trim(strings.Join(strings.Fields(fmt.Sprint(tc.sh1)), ","), "[]"),
				), func(t *testing.T) {

					t1 := ctx.(*Context).FromBytes(dtype, quantData[dtype], tc.sh1...)
					slog.Debug("Created", "t1", t1)
					defer func() {
						e := recover()
						if e != nil {
							t.Fatalf("hit unexpected panic: %v", e)
						}
					}()
					t2 := t1.Permute(ctx, p1...)
					slog.Debug("Permuted", "t2", t2)

					expStride := expStride(tc.st1, p1)
					sh := make([]int, len(tc.sh1))
					for i := range sh {
						sh[i] = t2.Stride(i)
					}
					if !reflect.DeepEqual(sh, expStride) {
						t.Fatalf("incorrect result\ngot %v\nexpected %v", sh, expStride)
					}
				})
			}
		}
	}
}

func TestQuantAsStrided(t *testing.T) {
	/* Equivalent pytorch for building expVals
	import torch
	import math
	shape = (8,32)
	t1 = torch.arange(math.prod(shape)).view(shape)
	print("t1:", t1, "shape:", t1.shape, "stride:", t1.stride())
	t2 = torch.as_strided(t1, (2,64), (128,1), 32)
	print("t2:", t2, "shape:", t2.shape, "stride:", t2.stride())
	*/
	type testCase struct {
		sh1      []int // Initial shape
		sh2      []int // view shape
		st2      []int // view stride
		offset   int   // view offset
		expVals  []float32
		expError string
	}
	testCases := []testCase{
		{ // 2d -> 2d, combine rows, with gap
			sh1:    []int{8, 32},
			offset: 32,
			sh2:    []int{2, 64},
			st2:    []int{128, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
				46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
				60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
				74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
				88, 89, 90, 91, 92, 93, 94, 95,
				160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
				174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
				188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201,
				202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215,
				216, 217, 218, 219, 220, 221, 222, 223,
			},
		},

		{ // 2d -> 3d, 2 rows no gaps
			sh1:    []int{8, 32},
			offset: 32,
			sh2:    []int{1, 2, 32},
			st2:    []int{1, 32, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
				81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
			},
		},
		{ // 2d -> 3d, 2 rows, with gap
			sh1:    []int{8, 32},
			offset: 32,
			sh2:    []int{1, 2, 32},
			st2:    []int{1, 64, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
				110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
				124, 125, 126, 127,
			},
		},
		{ // 2d -> 4d, 2 rows no gaps
			sh1:    []int{8, 32},
			offset: 32,
			sh2:    []int{1, 1, 2, 32},
			st2:    []int{1, 32, 32, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
				81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
			},
		},

		{
			sh1:    []int{2, 4, 32},
			offset: 128,
			sh2:    []int{1, 2, 32},
			st2:    []int{128, 32, 1},
			expVals: []float32{
				128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
				142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
				156, 157, 158, 159,
				160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
				174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
				188, 189, 190, 191,
			},
		},
		{ // reshaping 4,32 -> 2,64
			sh1:    []int{2, 4, 32},
			offset: 64,
			sh2:    []int{1, 2, 64},
			st2:    []int{128, 64, 1},
			expVals: []float32{
				64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
				78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
				92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
				106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
				120, 121, 122, 123, 124, 125, 126, 127,
				128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
				142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
				156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
				170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
				184, 185, 186, 187, 188, 189, 190, 191,
			},
		},
		{ // reshaping 4,32 -> 2,64 with overlap due to stride
			sh1:    []int{2, 4, 32},
			offset: 64,
			sh2:    []int{1, 2, 64},
			st2:    []int{128, 32, 1},
			expVals: []float32{
				64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
				78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
				92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
				106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
				120, 121, 122, 123, 124, 125, 126, 127,
				96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
				110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
				124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
				138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151,
				152, 153, 154, 155, 156, 157, 158, 159,
			},
		},

		{
			sh1: []int{2, 4, 32},
			sh2: []int{1, 2, 32},
			st2: []int{128, 32, 1},
			expVals: []float32{
				128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
				142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
				156, 157, 158, 159,
				160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
				174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
				188, 189, 190, 191,
			},
			offset: 128,
		},
		{ // 2 rows, skipping a row
			sh1:    []int{4, 2, 32},
			offset: 128,
			sh2:    []int{1, 2, 32},
			st2:    []int{1, 64, 1},
			expVals: []float32{
				128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
				142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
				156, 157, 158, 159,
				192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205,
				206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
				220, 221, 222, 223,
			},
		},
		{ // 3d->2d, 2 rows, skipping a row
			sh1:    []int{4, 2, 32},
			offset: 128,
			sh2:    []int{2, 32},
			st2:    []int{64, 1},
			expVals: []float32{
				128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
				142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
				156, 157, 158, 159,
				192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205,
				206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
				220, 221, 222, 223,
			},
		},

		{ // 4d->4d, 2 rows, no skip or overlap
			sh1:    []int{2, 2, 2, 32},
			offset: 32,
			sh2:    []int{1, 2, 2, 32},
			st2:    []int{1, 64, 32, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
				46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
				60, 61, 62, 63,
				64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
				78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
				92, 93, 94, 95,
				96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
				110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
				124, 125, 126, 127,
				128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
				142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
				156, 157, 158, 159,
			},
		},
		{ // 4d->4d, 2 rows, skip a row
			sh1:    []int{2, 2, 2, 32},
			offset: 32,
			sh2:    []int{1, 2, 2, 32},
			st2:    []int{1, 128, 32, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
				46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
				60, 61, 62, 63,
				64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
				78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
				92, 93, 94, 95,
				160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
				174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
				188, 189, 190, 191,
				192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205,
				206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
				220, 221, 222, 223,
			},
		},
		{ // 4d->3d, no skip or overlap
			sh1:    []int{2, 2, 2, 32},
			offset: 32,
			sh2:    []int{2, 2, 32},
			st2:    []int{64, 32, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
				46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
				60, 61, 62, 63,
				64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
				78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
				92, 93, 94, 95,
				96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
				110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
				124, 125, 126, 127,
				128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
				142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
				156, 157, 158, 159,
			},
		},
		{ // 4d->2d, no skip or overlap
			sh1:    []int{2, 2, 2, 32},
			offset: 32,
			sh2:    []int{4, 32},
			st2:    []int{32, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
				46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
				60, 61, 62, 63,
				64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
				78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
				92, 93, 94, 95,
				96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
				110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
				124, 125, 126, 127,
				128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
				142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
				156, 157, 158, 159,
			},
		},
		{ // 4d->2d, skip rows
			sh1:    []int{2, 2, 2, 32},
			offset: 32,
			sh2:    []int{4, 32},
			st2:    []int{64, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
				46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
				60, 61, 62, 63,
				96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
				110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
				124, 125, 126, 127,
				160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
				174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
				188, 189, 190, 191,
				224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
				238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251,
				252, 253, 254, 255,
			},
		},

		{ // 3d->2d, 1/2 row, skip a row
			sh1:    []int{4, 2, 32},
			offset: 96,
			sh2:    []int{2, 8},
			st2:    []int{16, 1},
			expVals: []float32{ // if supported, this would be the result
				96, 97, 98, 99, 100, 101, 102, 103,
				112, 113, 114, 115, 116, 117, 118, 119,
			},
			expError: "whole row",
		},
		{ // 2d -> 4d, 2 rows no gaps, single dims
			sh1:      []int{8, 32},
			offset:   32,
			sh2:      []int{1, 2, 1, 32},
			st2:      []int{1, 32, 32, 1},
			expError: "intermixed single dimensions",
			expVals: []float32{ // if supported
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
				81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
			},
		},
		{ // 2d -> 4d, 2 rows no gaps, single dims
			sh1:      []int{8, 32},
			offset:   32,
			sh2:      []int{2, 1, 1, 32},
			st2:      []int{32, 1, 32, 1},
			expError: "intermixed single dimensions",
			expVals: []float32{ // if supported
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
				81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
			},
		},

		// TODO more error cases

	}

	be := newTestBackend(100)
	ctx := newTestContext(be, 100).Input()

	for _, dtype := range []ml.DType{ml.DTypeQ40, ml.DTypeQ80} {
		for _, tc := range testCases {
			t.Run(fmt.Sprintf("%s/sh1:%s_sh2:%s_st2:%s_offset:%d",
				dtype,
				strings.Trim(strings.Join(strings.Fields(fmt.Sprint(tc.sh1)), ","), "[]"),
				strings.Trim(strings.Join(strings.Fields(fmt.Sprint(tc.sh2)), ","), "[]"),
				strings.Trim(strings.Join(strings.Fields(fmt.Sprint(tc.st2)), ","), "[]"),
				tc.offset,
			), func(t *testing.T) {

				t1 := ctx.(*Context).FromBytes(dtype, quantData[dtype], tc.sh1...)
				slog.Debug("Created", "t1", t1)
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
				ctx.Forward(t1)
				dump(ml.Dump(ctx, t1))

				t2 := t1.AsStrided(ctx, tc.sh2, tc.st2, tc.offset)
				slog.Debug("AsStrided", "t2", t2)
				if tc.expError != "" {
					t.Fatalf("expected error %s but AsStrided did not panic", tc.expError)
				}
				dump(fmt.Sprintf("Dumping t2 result shape:%v\n", t2.Shape()))
				dump(ml.Dump(ctx, t2))
				if !reflect.DeepEqual(t2.Shape(), tc.sh2) {
					t.Errorf("incorrect shape result\ngot %v\nexpected %v", t2.Shape(), tc.sh2)
				}
				// Verify data is similar to expected
				t3 := ctx.Empty(ml.DTypeF32, t2.Shape()...)
				t3 = t2.Copy(ctx, t3)
				ctx.Forward(t3).Compute(t3)
				f32s := t3.Floats()
				if len(f32s) != len(tc.expVals) {
					t.Errorf("incorrect data length result\ngot %v\nexpected %v", len(f32s), len(tc.expVals))
				}
				sim := cosineSimilarity(f32s, tc.expVals)
				if sim < 0.999 || math.IsNaN(float64(sim)) { // account for quantization drift
					t.Errorf("too low cosine similarity: %f", sim)
				}
				slog.Debug("Cosine", "similarity", sim)

				st := make([]int, len(tc.sh2))
				for i := range st {
					st[i] = t2.Stride(i)
				}
			})
		}
	}
}

func TestPermutedQuantAsStrided(t *testing.T) {
	/* Equivalent to the following pytorch - all expected results derived from this

	import torch
	t1 = torch.arange(12).view(3,4).permuted((1,0))
	print(t1)
	t2 = torch.as_strided(t1, (2,2), (1,2), 1)
	print(t2)
	*/

	type testCase struct {
		sh1      []int // Initial shape
		p1       []int // Permute call on initial tensor
		sh2      []int // view shape
		st2      []int // view stride
		offset   int   // view offset
		expVals  []float32
		expError string
	}
	testCases := []testCase{
		{ // 2d -> 2d
			sh1:    []int{8, 32},
			p1:     []int{1, 0, 2, 3}, // underlying 0,2,1,3
			offset: 32,
			sh2:    []int{2, 32},
			st2:    []int{32, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
				81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
			},
		},
		{ // 2d -> 2d skip row
			sh1:    []int{8, 32},
			p1:     []int{1, 0, 2, 3}, // underlying 0,2,1,3
			offset: 32,
			sh2:    []int{2, 32},
			st2:    []int{64, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
				110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
				124, 125, 126, 127,
			},
		},

		{ // 2d -> 3d
			sh1:    []int{8, 32},
			p1:     []int{1, 0, 2, 3}, // underlying 0,2,1,3
			offset: 32,
			sh2:    []int{1, 2, 32},
			st2:    []int{1, 32, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
				81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
			},
		},
		{ // 2d -> 3d skip row
			sh1:    []int{8, 32},
			p1:     []int{1, 0, 2, 3}, // underlying 0,2,1,3
			offset: 32,
			sh2:    []int{1, 2, 32},
			st2:    []int{1, 64, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
				110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
				124, 125, 126, 127,
			},
		},
		{ // 2d -> 4d
			sh1:    []int{8, 32},
			p1:     []int{1, 0, 2, 3}, // underlying 0,2,1,3
			offset: 32,
			sh2:    []int{1, 2, 2, 32},
			st2:    []int{1, 64, 32, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
				81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
				96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
				110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
				124, 125, 126, 127,
				128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
				142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
				156, 157, 158, 159,
			},
		},
		{ // 2d -> 4d skip rows
			sh1:    []int{8, 32},
			p1:     []int{1, 0, 2, 3}, // underlying 0,2,1,3
			offset: 32,
			sh2:    []int{1, 2, 2, 32},
			st2:    []int{1, 128, 64, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
				110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
				124, 125, 126, 127,
				160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
				174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
				188, 189, 190, 191,
				224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
				238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251,
				252, 253, 254, 255,
			},
		},

		{ // 3d -> 2d
			sh1:    []int{2, 4, 32},
			p1:     []int{1, 0, 2, 3}, // underlying 0,2,1,3
			offset: 32,
			sh2:    []int{2, 32},
			st2:    []int{32, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
				81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
			},
		},
		{ // 3d -> 2d skip row
			sh1:    []int{2, 4, 32},
			p1:     []int{1, 0, 2, 3}, // underlying 0,2,1,3
			offset: 32,
			sh2:    []int{2, 32},
			st2:    []int{64, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
				110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
				124, 125, 126, 127,
			},
		},
		{ // 3d -> 2d skip row
			sh1:    []int{2, 4, 32},
			p1:     []int{2, 1, 0, 3},
			offset: 32,
			sh2:    []int{2, 32},
			st2:    []int{64, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
				110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
				124, 125, 126, 127,
			},
		},

		{ // 3d -> 3d
			sh1:    []int{2, 4, 32},
			p1:     []int{1, 0, 2, 3}, // underlying 0,2,1,3
			offset: 32,
			sh2:    []int{1, 2, 32},
			st2:    []int{1, 32, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
				81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
			},
		},
		{ // 3d -> 3d skip row
			sh1:    []int{2, 4, 32},
			p1:     []int{1, 0, 2, 3}, // underlying 0,2,1,3
			offset: 32,
			sh2:    []int{1, 2, 32},
			st2:    []int{1, 64, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
				110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
				124, 125, 126, 127,
			},
		},

		{ // 3d -> 4d
			sh1:    []int{2, 4, 32},
			p1:     []int{1, 0, 2, 3}, // underlying 0,2,1,3
			offset: 32,
			sh2:    []int{1, 2, 2, 32},
			st2:    []int{1, 64, 32, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
				81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
				96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
				110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
				124, 125, 126, 127,
				128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
				142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
				156, 157, 158, 159,
			},
		},
		{ // 3d -> 4d skip row
			sh1:    []int{2, 4, 32},
			p1:     []int{1, 0, 2, 3}, // underlying 0,2,1,3
			offset: 32,
			sh2:    []int{1, 2, 2, 32},
			st2:    []int{1, 128, 32, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
				81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
				160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
				174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
				188, 189, 190, 191,
				192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205,
				206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
				220, 221, 222, 223,
			},
		},

		{ // 4d -> 2d
			sh1:    []int{2, 2, 2, 32},
			p1:     []int{2, 0, 1, 3},
			offset: 32,
			sh2:    []int{3, 32},
			st2:    []int{32, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
				81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
				96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
				110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
				124, 125, 126, 127,
			},
		},
		{ // 4d -> 2d skip row
			sh1:    []int{2, 2, 2, 32},
			p1:     []int{2, 0, 1, 3},
			offset: 32,
			sh2:    []int{3, 32},
			st2:    []int{64, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
				110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
				124, 125, 126, 127,
				160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
				174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
				188, 189, 190, 191,
			},
		},
		{ // 4d -> 3d
			sh1:    []int{2, 2, 2, 32},
			p1:     []int{2, 0, 1, 3},
			offset: 32,
			sh2:    []int{2, 2, 32},
			st2:    []int{64, 32, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
				81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
				96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
				110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
				124, 125, 126, 127,
				128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
				142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
				156, 157, 158, 159,
			},
		},

		{ // 4d -> 3d skip rows
			sh1:    []int{2, 2, 2, 32},
			p1:     []int{2, 0, 1, 3},
			offset: 32,
			sh2:    []int{2, 2, 32},
			st2:    []int{128, 64, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
				110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
				124, 125, 126, 127,
				160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
				174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
				188, 189, 190, 191,
				224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
				238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251,
				252, 253, 254, 255,
			},
		},
		{ // 4d -> 4d
			sh1:    []int{2, 2, 2, 32},
			p1:     []int{2, 0, 1, 3},
			offset: 32,
			sh2:    []int{1, 2, 2, 32},
			st2:    []int{1, 64, 32, 1},
			expVals: []float32{
				32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
				64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
				81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
				96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
				110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
				124, 125, 126, 127,
				128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
				142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
				156, 157, 158, 159,
			},
		},
	}

	be := newTestBackend(100)
	ctx := newTestContext(be, 100).Input()

	for _, dtype := range []ml.DType{ml.DTypeQ40, ml.DTypeQ80} {
		for _, tc := range testCases {
			t.Run(fmt.Sprintf("%s/sh1:%s_sh2:p1:%s_%s_st2:%s_offset:%d",
				dtype,
				strings.Trim(strings.Join(strings.Fields(fmt.Sprint(tc.sh1)), ","), "[]"),
				strings.Trim(strings.Join(strings.Fields(fmt.Sprint(tc.p1)), ","), "[]"),
				strings.Trim(strings.Join(strings.Fields(fmt.Sprint(tc.sh2)), ","), "[]"),
				strings.Trim(strings.Join(strings.Fields(fmt.Sprint(tc.st2)), ","), "[]"),
				tc.offset,
			), func(t *testing.T) {

				t1 := ctx.(*Context).FromBytes(dtype, quantData[dtype], tc.sh1...)
				slog.Debug("Created", "t1", t1)
				t2 := t1.Permute(ctx, tc.p1...)
				slog.Debug("Permuted", "t2", t2)
				ctx.Forward(t2)
				dump(ml.Dump(ctx, t2))
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
				t3 := t2.AsStrided(ctx, tc.sh2, tc.st2, tc.offset) //.Contiguous(ctx)
				slog.Debug("AsStrided", "t3", t3)
				ctx.Forward(t3)
				if tc.expError != "" {
					t.Fatalf("expected error %s but AsStrided did not panic", tc.expError)
				}
				dump(fmt.Sprintf("Dumping t3 result shape:%v\n", t3.Shape()))
				dump(ml.Dump(ctx, t3))
				if !reflect.DeepEqual(t3.Shape(), tc.sh2) {
					t.Errorf("incorrect shape result\ngot %v\nexpected %v", t3.Shape(), tc.sh2)
				}
				// Verify data is similar to expected
				t4 := ctx.Empty(ml.DTypeF32, t3.Shape()...)
				t4 = t3.Copy(ctx, t4)
				ctx.Forward(t4).Compute(t4)
				f32s := t4.Floats()
				if len(f32s) != len(tc.expVals) {
					t.Errorf("incorrect data length result\ngot %v\nexpected %v", len(f32s), len(tc.expVals))
				}
				sim := cosineSimilarity(f32s, tc.expVals)
				if sim < 0.999 || math.IsNaN(float64(sim)) { // account for quantization drift
					t.Errorf("too low cosine similarity: %f", sim)
				}
				slog.Debug("Cosine", "similarity", sim)
			})
		}
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

func dotProduct[V float32 | float64](v1, v2 []V) V {
	var result V = 0
	for i := 0; i < len(v1); i++ {
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

func expStride(stride, permute []int) []int {
	// Derive expected stride
	expStride := make([]int, len(stride))
	for i := range expStride {
		//  permute(2, 0, 1) => (0 comes from 2, 1 comes from 0, 2 comes from 1)
		expStride[i] = stride[permute[i]]
	}
	return expStride
}

// arange 256 quantized with gguf.quantize(<data>, <type>)
var quantData = map[ml.DType][]uint8{
	ml.DTypeQ40: []uint8{
		192, 195, 72, 72, 55, 55, 55, 55, 38, 38, 38, 38, 21, 21, 21, 21, 4, 4,
		224, 199, 36, 36, 36, 36, 19, 19, 19, 19, 19, 19, 19, 19, 2, 2, 2, 2,
		240, 201, 19, 19, 18, 18, 18, 18, 18, 18, 18, 18, 2, 2, 2, 2, 1, 1,
		240, 203, 18, 18, 18, 18, 18, 18, 18, 18, 1, 1, 1, 1, 1, 1, 1, 1,
		248, 204, 18, 18, 17, 17, 17, 17, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		248, 205, 17, 17, 17, 17, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		248, 206, 17, 17, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		248, 207, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	},

	ml.DTypeQ80: []uint8{
		208, 51, 0, 4, 8, 12, 16, 20, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 66, 70, 74, 78, 82, 86,
		90, 94, 98, 102, 107, 111, 115, 119, 123, 127, 240, 55, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83,
		85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127,
		252, 57, 86, 87, 88, 90, 91, 92, 94, 95, 96, 98, 99, 100, 102, 103, 104, 106, 107, 108, 110, 111, 112, 114,
		115, 116, 118, 119, 120, 122, 123, 124, 126, 127, 0, 60, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
		106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
		2, 61, 102, 103, 104, 105, 105, 106, 107, 108, 109, 109, 110, 111, 112, 113, 113, 114, 115, 116, 117, 117, 118, 119,
		120, 121, 121, 122, 123, 124, 125, 125, 126, 127, 4, 62, 106, 107, 108, 108, 109, 110, 110, 111, 112, 112,
		113, 114, 114, 115, 116, 116, 117, 118, 118, 119, 120, 120, 121, 122, 122, 123, 124, 124, 125, 126, 126, 127,
		6, 63, 109, 110, 110, 111, 112, 112, 113, 113, 114, 114, 115, 116, 116, 117, 117, 118, 118, 119, 120, 120, 121, 121,
		122, 122, 123, 124, 124, 125, 125, 126, 126, 127, 4, 64, 112, 112, 113, 113, 114, 114, 115, 115, 116, 116,
		117, 117, 118, 118, 119, 119, 120, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 126, 126, 127, 127,
	},
}

var permutes = map[int][][]int{
	2: [][]int{
		{1, 0, 2, 3},
	},
	3: [][]int{
		{0, 2, 1, 3},
		{1, 0, 2, 3},
		{1, 2, 0, 3},
		{2, 0, 1, 3},
		{2, 1, 0, 3},
	},
	4: [][]int{
		{0, 1, 3, 2},
		{0, 2, 1, 3},
		{0, 2, 3, 1},
		{0, 3, 1, 2},
		{0, 3, 2, 1},
		{1, 0, 2, 3},
		{1, 0, 3, 2},
		{1, 2, 0, 3},
		{1, 2, 3, 0},
		{1, 3, 0, 2},
		{1, 3, 2, 0},
		{2, 0, 1, 3},
		{2, 0, 3, 1},
		{2, 1, 0, 3},
		{2, 1, 3, 0},
		{2, 3, 0, 1},
		{2, 3, 1, 0},
		{3, 0, 1, 2},
		{3, 0, 2, 1},
		{3, 1, 0, 2},
		{3, 1, 2, 0},
		{3, 2, 0, 1},
		{3, 2, 1, 0},
	},
}
