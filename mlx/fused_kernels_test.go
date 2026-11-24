package mlx

import (
	"fmt"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func requireMetal(t *mlxthreadtest.T) {
	if !MetalIsAvailable() {
		t.Skip("requires Metal")
	}
}

func TestFusedShortMatmulSoftcap(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		requireMetal(t)
		const width, columns, cap = 2304, 40000, 30
		w := patternArray(DTypeBFloat16, []int{columns, width}, 0, 0.02, 7, 97)
		for _, rows := range []int{1, 4, 8} {
			x := patternArray(DTypeBFloat16, []int{rows, width}, 0.1, 0.03, 11, 89)
			got := ShortMatmulSoftcap(x, w, cap)
			if got == nil || got.DType() != DTypeFloat32 {
				t.Fatal("kernel did not run in FP32")
			}
			plain := ShortMatmul(x.AsType(DTypeFloat32), w)
			want := MulScalar(DivScalar(plain, cap).Tanh(), cap)
			if err := requireRelative(fmt.Sprintf("rows %d", rows), got, want, 1e-5); err != nil {
				t.Fatal(err)
			}
		}
	})
}

func TestFusedShortMatmulRows(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		requireMetal(t)
		// Eight rows in one launch match each row computed alone, for narrow
		// and wide products; nine rows are declined.
		for _, columns := range []int{128, 2304} {
			for _, weightType := range []DType{DTypeFloat32, DTypeBFloat16} {
				const rows, width = 8, 2304
				w := patternArray(weightType, []int{columns, width}, 0, 0.01, 7, 97)
				x := patternArray(DTypeBFloat16, []int{rows, width}, 0.1, 0.02, 11, 89)
				if ShortMatmul(patternArray(DTypeBFloat16, []int{rows + 1, width}, 0, 0.01, 3, 7), w) != nil {
					t.Fatal("selected the short product for nine rows")
				}
				all := ShortMatmul(x, w).AsType(DTypeFloat32)
				for m := range rows {
					alone := ShortMatmul(x.Slice(Slice(m, m+1), Slice()), w).AsType(DTypeFloat32)
					if err := requireExact("row", alone, all.Slice(Slice(m, m+1), Slice())); err != nil {
						t.Fatalf("%d columns %v: %v", columns, weightType, err)
					}
				}
			}
		}
	})
}

// shortConvolutionGraph is ShortConvolution with fixed taps in graph ops: the
// FP32 tap products summed oldest first, plus the input, times scale.
func shortConvolutionGraph(state, x, weight *Array, scale float32) (y, next *Array) {
	tokens, taps := x.Dim(1), weight.Dim(0)
	window := Concatenate([]*Array{state, x}, 1)
	var sum *Array
	for i := range taps {
		lagged := window.Slice(Slice(), Slice(i, i+tokens), Slice()).AsType(DTypeFloat32)
		product := Mul(lagged, weight.Slice(Slice(i, i+1), Slice()).AsType(DTypeFloat32))
		if sum == nil {
			sum = product
		} else {
			sum = Add(sum, product)
		}
	}
	y = MulScalar(Add(sum, x.AsType(DTypeFloat32)), scale).AsType(x.DType())
	return y, Contiguous(window.Slice(Slice(), Slice(tokens, tokens+taps-1), Slice()), false)
}

// Decode reads x and the residual as column views of wider products. Both
// kernel variants must match their contiguous copies and the graph
// convolution followed by an FP32 residual add, bit for bit.
func TestFusedShortConvolutionAdd(t *testing.T) {
	const channels, taps, pad, scale = 2304, 4, 32, 0.4472136
	for _, tc := range []struct {
		rows, tokens int
		strided      bool
	}{
		{1, 1, false},
		{2, 4, true},
		{1, 17, false},
	} {
		t.Run(fmt.Sprintf("%d_positions", tc.rows*tc.tokens), func(t *testing.T) {
			withMLXThread(t, func(t *mlxthreadtest.T) {
				requireMetal(t)
				if strided := shortConvolutionConfig(tc.rows, tc.tokens) == shortConvolutionStrided; strided != tc.strided {
					t.Fatalf("strided variant selected: %t, want %t", strided, tc.strided)
				}
				columns := func(dtype DType, bias, step float32, stride, modulus int) *Array {
					wide := patternArray(dtype, []int{tc.rows, tc.tokens, channels + 2*pad}, bias, step, stride, modulus)
					return wide.Slice(Slice(), Slice(), Slice(pad, pad+channels))
				}
				state := patternArray(DTypeBFloat16, []int{tc.rows, taps - 1, channels}, 0, 0.02, 5, 101)
				weight := patternArray(DTypeBFloat16, []int{taps, channels}, 0.1, 0.01, 3, 107)
				x := columns(DTypeBFloat16, 0.01, 0.03, 7, 103)
				y, next := shortConvolutionGraph(state, x, weight, scale)
				gotY, gotNext := ShortConvolution(state, x, weight, nil, 0, scale)
				if gotY == nil {
					t.Fatal("convolution kernel did not run")
				}
				if err := requireExact("convolution", gotY, y); err != nil {
					t.Fatal(err)
				}
				if err := requireExact("convolution state", gotNext, next); err != nil {
					t.Fatal(err)
				}
				for _, residualType := range []DType{DTypeBFloat16, DTypeFloat32} {
					residual := columns(residualType, -0.2, 0.05, 13, 109)
					for _, dtype := range []DType{DTypeBFloat16, DTypeFloat32} {
						sum, sumNext := ShortConvolutionAdd(state, x, weight, residual, scale, dtype)
						copied, copiedNext := ShortConvolutionAdd(state, Contiguous(x, false), weight, Contiguous(residual, false), scale, dtype)
						if sum == nil || copied == nil {
							t.Fatal("kernel did not run")
						}
						want := Add(residual.AsType(DTypeFloat32), y.AsType(DTypeFloat32)).AsType(dtype)
						for _, check := range []struct {
							label     string
							got, want *Array
						}{
							{"sum of views", sum, copied},
							{"state of views", sumNext, copiedNext},
							{"sum", sum, want},
							{"state", sumNext, next},
						} {
							label := fmt.Sprintf("%v residual, %v sum: %s", residualType, dtype, check.label)
							if err := requireExact(label, check.got, check.want); err != nil {
								t.Fatal(err)
							}
						}
					}
				}
			})
		})
	}
}

// controllerGraph computes the looped-layer controller in FP32 graph ops for
// inputs [rows, inputs, width] and messages [rows, 4, width].
func controllerGraph(input, message, down, gates *Array, inputs, i int) (compressed, out *Array) {
	rows, width := message.Dim(0), message.Dim(2)
	in := input.Slice(Slice(), Slice(min(i, inputs-1), min(i, inputs-1)+1), Slice()).Reshape(rows, width)
	msg := message.Slice(Slice(), Slice(i, i+1), Slice()).Reshape(rows, width)
	paired := Concatenate([]*Array{RMSNormFn(in, nil, 1e-6), RMSNormFn(msg, nil, 1e-6)}, -1)
	d := down.Slice(Slice(i, i+1), Slice(), Slice()).Squeeze(0).AsType(DTypeFloat32)
	compressed = paired.Matmul(d.Transpose(1, 0))
	expansion := compressed.Matmul(gates.Slice(Slice(i, i+1), Slice(), Slice()).Squeeze(0))
	gain := expansion.Slice(Slice(), Slice(0, width)).Tanh()
	forget := expansion.Slice(Slice(), Slice(width, 2*width)).Tanh()
	out = Add(Mul(in, AddScalar(MulScalar(forget, -1), 1)), Mul(msg, gain))
	return compressed, out
}

func TestFusedLoopControllers(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		requireMetal(t)
		if loopControllerDownConfig(kernelArchitecture()) == 0 {
			t.Skip("the controller down kernel does not run on this device")
		}
		const width, rank, count = 2304, 64, 3
		down := patternArray(DTypeBFloat16, []int{count, rank, 2 * width}, 0, 0.002, 7, 211)
		gates := patternArray(DTypeFloat32, []int{count, rank, 2 * width}, 0, 0.003, 11, 223)
		norms := patternArray(DTypeFloat32, []int{count, width}, 1, 0.01, 17, 61)
		stack := patternArray(DTypeBFloat16, []int{300, width}, 0, 0.01, 19, 227)
		ends := [2]int{120, 210}
		// 12 rows leave the second tile of 8 partly filled.
		for _, rows := range []int{1, 4, 12, 16, 24} {
			for _, inputs := range []int{1, 3} {
				input := patternArray(DTypeFloat32, []int{rows, inputs, width}, 0.02, 0.05, 13, 229)
				message := patternArray(DTypeFloat32, []int{rows, 4, width}, -0.03, 0.04, 23, 233)
				compressed := LoopControllerDown(input, inputs, message, 0, count, down, 1e-6)
				if compressed == nil {
					t.Fatal("down kernel did not run")
				}
				plain := LoopControllerGate(input, inputs, message, 0, compressed, gates)
				scaled, squares := LoopControllerGateNorm(input, inputs, message, 0, compressed, gates, norms)
				if plain == nil || scaled == nil {
					t.Fatal("gate kernel did not run")
				}
				projected := GroupedShortMatmul(scaled, rows, width, rows*width, ends, stack, squares, 1e-6)
				if projected == nil {
					t.Fatal("grouped product did not run")
				}
				var groups []*Array
				for i := range count {
					label := fmt.Sprintf("rows %d inputs %d controller %d", rows, inputs, i)
					wantCompressed, wantOut := controllerGraph(input, message, down, gates, inputs, i)
					got := compressed.Slice(Slice(i, i+1), Slice(), Slice()).Squeeze(0)
					// The kernel rounds the compressed values to BF16.
					if err := requireRelative(label+" compressed", got, wantCompressed, 1.0/64); err != nil {
						t.Fatal(err)
					}
					gotOut := plain.Slice(Slice(i, i+1), Slice(), Slice()).Squeeze(0)
					if err := requireRelative(label+" gate", gotOut, wantOut, 1.0/64); err != nil {
						t.Fatal(err)
					}
					normalized := RMSNormFn(gotOut, norms.Slice(Slice(i, i+1), Slice()).Reshape(width), 1e-6)
					bounds := []int{0, ends[0], ends[1], stack.Dim(0)}
					w := stack.Slice(Slice(bounds[i], bounds[i+1]), Slice()).AsType(DTypeFloat32)
					groups = append(groups, normalized.Matmul(w.Transpose(1, 0)))
				}
				want := Concatenate(groups, 1)
				if err := requireRelative(fmt.Sprintf("rows %d inputs %d projection", rows, inputs), projected, want, 1.0/64); err != nil {
					t.Fatal(err)
				}
			}
		}
	})
}

// The residual mixer returns its queries as a transposed view; the controllers
// read such views in place and match their contiguous copies bit for bit.
func TestFusedLoopControllersReadViews(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		requireMetal(t)
		if loopControllerDownConfig(kernelArchitecture()) == 0 {
			t.Skip("the controller down kernel does not run on this device")
		}
		const width, rank = 2304, 64
		down := patternArray(DTypeBFloat16, []int{3, rank, 2 * width}, 0, 0.002, 7, 211)
		gates := patternArray(DTypeFloat32, []int{3, rank, 2 * width}, 0, 0.003, 11, 223)
		norms := patternArray(DTypeFloat32, []int{3, width}, 1, 0.01, 17, 61)
		// queryMajor returns [lead..., slots, width] as a view of [slots, lead..., width].
		queryMajor := func(lead []int, slots, seed int) *Array {
			x := patternArray(DTypeFloat32, append(append([]int{slots}, lead...), width), 0.01, 0.04, seed, 239)
			if len(lead) == 1 {
				return x.Transpose(1, 0, 2)
			}
			return x.Transpose(1, 2, 0, 3)
		}
		for _, lead := range [][]int{{5}, {2, 3}} {
			message := queryMajor(lead, 4, 31)
			for _, tc := range []struct {
				input         *Array
				inputs, first int
			}{
				{queryMajor(lead, 3, 29), 3, 0},
				{patternArray(DTypeFloat32, append(slices.Clone(lead), width), 0.01, 0.04, 37, 241), 1, 1},
			} {
				count := tc.inputs
				d, g, n := down.Slice(Slice(0, count), Slice(), Slice()), gates.Slice(Slice(0, count), Slice(), Slice()), norms.Slice(Slice(0, count), Slice())
				label := fmt.Sprintf("lead %v inputs %d", lead, tc.inputs)
				view := LoopControllerDown(tc.input, tc.inputs, message, tc.first, count, d, 1e-6)
				copied := LoopControllerDown(Contiguous(tc.input, false), tc.inputs, Contiguous(message, false), tc.first, count, d, 1e-6)
				if view == nil || copied == nil {
					t.Fatalf("%s: down kernel did not run", label)
				}
				if err := requireExact(label+" compressed", view, copied); err != nil {
					t.Fatal(err)
				}
				gotOut, gotSquares := LoopControllerGateNorm(tc.input, tc.inputs, message, tc.first, view, g, n)
				wantOut, wantSquares := LoopControllerGateNorm(Contiguous(tc.input, false), tc.inputs, Contiguous(message, false), tc.first, copied, g, n)
				if gotOut == nil || wantOut == nil {
					t.Fatalf("%s: gate kernel did not run", label)
				}
				if err := requireExact(label+" gated", gotOut, wantOut); err != nil {
					t.Fatal(err)
				}
				if err := requireExact(label+" squares", gotSquares, wantSquares); err != nil {
					t.Fatal(err)
				}
			}
		}
	})
}

// Layouts the fast loads cannot take still match their contiguous copies bit
// for bit.
func TestFusedLoopControllersReadAnyLayout(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		requireMetal(t)
		if loopControllerDownConfig(kernelArchitecture()) == 0 {
			t.Skip("the controller down kernel does not run on this device")
		}
		const width, rank, rows, count = 2304, 64, 6, 2
		down := patternArray(DTypeBFloat16, []int{count, rank, 2 * width}, 0, 0.002, 7, 211)
		gates := patternArray(DTypeFloat32, []int{count, rank, 2 * width}, 0, 0.003, 11, 223)
		norms := patternArray(DTypeFloat32, []int{count, width}, 1, 0.01, 17, 61)
		compressed := patternArray(DTypeFloat32, []int{count, rows, rank}, 0, 0.05, 5, 37)
		input := patternArray(DTypeFloat32, []int{rows, width}, 0.02, 0.05, 13, 229)
		message := patternArray(DTypeFloat32, []int{rows, 4, width}, -0.03, 0.04, 23, 233)
		// stored returns x as a view of a copy transposed by perm, its own inverse.
		stored := func(x *Array, perm ...int) *Array {
			return Contiguous(x.Transpose(perm...), false).Transpose(perm...)
		}
		type operands struct{ input, message, down, gates, norms, compressed *Array }
		for _, tc := range []struct {
			name string
			view operands
		}{
			{"query_major_message", operands{input, stored(message, 1, 0, 2), down, gates, norms, compressed}},
			{"strided_width", operands{input, stored(message, 0, 2, 1), down, gates, norms, compressed}},
			{"unmerged_rows", operands{input, stored(message.Reshape(2, 3, 4, width), 1, 0, 2, 3), down, gates, norms, compressed}},
			// Rows of 2306 values are not 16-byte aligned for the down kernel's vector loads.
			{"unaligned_input_rows", operands{patternArray(DTypeFloat32, []int{rows, width + 2}, 0.02, 0.05, 13, 229).Slice(Slice(), Slice(0, width)), message, down, gates, norms, compressed}},
			{"strided_weights", operands{input, message, stored(down, 0, 2, 1), stored(gates, 0, 2, 1), stored(norms, 1, 0), stored(compressed, 0, 2, 1)}},
		} {
			v := tc.view
			d := operands{
				Contiguous(v.input, false), Contiguous(v.message, false), Contiguous(v.down, false),
				Contiguous(v.gates, false), Contiguous(v.norms, false), Contiguous(v.compressed, false),
			}
			gotDown := LoopControllerDown(v.input, 1, v.message, 0, count, v.down, 1e-6)
			wantDown := LoopControllerDown(d.input, 1, d.message, 0, count, d.down, 1e-6)
			gotOut, gotSquares := LoopControllerGateNorm(v.input, 1, v.message, 0, v.compressed, v.gates, v.norms)
			wantOut, wantSquares := LoopControllerGateNorm(d.input, 1, d.message, 0, d.compressed, d.gates, d.norms)
			gotPlain := LoopControllerGate(v.input, 1, v.message, 0, v.compressed, v.gates)
			wantPlain := LoopControllerGate(d.input, 1, d.message, 0, d.compressed, d.gates)
			if gotDown == nil || wantDown == nil || gotOut == nil || wantOut == nil || gotPlain == nil || wantPlain == nil {
				t.Fatalf("%s: kernels did not run", tc.name)
			}
			for _, check := range []struct {
				label     string
				got, want *Array
			}{
				{"down", gotDown, wantDown},
				{"gate", gotOut, wantOut},
				{"squares", gotSquares, wantSquares},
				{"unscaled gate", gotPlain, wantPlain},
			} {
				if err := requireExact(tc.name+" "+check.label, check.got, check.want); err != nil {
					t.Fatal(err)
				}
			}
		}
	})
}

func TestFusedGroupedShortMatmulMatchesShortMatmul(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		requireMetal(t)
		const width, rows = 2304, 6
		stack := patternArray(DTypeBFloat16, []int{3000, width}, 0, 0.01, 19, 227)
		x := patternArray(DTypeBFloat16, []int{rows, 3, width}, 0.02, 0.05, 13, 229)
		ends := [2]int{1000, 2100}
		got := GroupedShortMatmul(x, rows, 3*width, width, ends, stack, nil, 0)
		if got == nil {
			t.Fatal("grouped product did not run")
		}
		bounds := []int{0, ends[0], ends[1], stack.Dim(0)}
		for g := range 3 {
			rowsOf := Contiguous(x.Slice(Slice(), Slice(g, g+1), Slice()).Reshape(rows, width), false)
			want := ShortMatmul(rowsOf, stack.Slice(Slice(bounds[g], bounds[g+1]), Slice()))
			if err := requireExact("group", got.Slice(Slice(), Slice(bounds[g], bounds[g+1])), want); err != nil {
				t.Fatalf("group %d: %v", g, err)
			}
		}
	})
}

// GroupedShortMatmul declines rows its tiles cannot address within x and
// squares that do not match its rows and width.
func TestFusedGroupedShortMatmulRejects(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		const width, rows, columns = 2304, 4, 3000
		weight := Zeros(DTypeBFloat16, columns, width)
		type call struct {
			x                            *Array
			rows, rowStride, groupStride int
			ends                         [2]int
			squares                      *Array
		}
		valid := call{Zeros(DTypeBFloat16, rows, 3, width), rows, 3 * width, width, [2]int{1000, 2100}, Zeros(DTypeFloat32, 3, rows, width/128)}
		run := func(c call) *Array {
			return GroupedShortMatmul(c.x, c.rows, c.rowStride, c.groupStride, c.ends, weight, c.squares, 1e-6)
		}
		if MetalIsAvailable() && run(valid) == nil {
			t.Fatal("valid inputs declined")
		}
		for _, tc := range []struct {
			name   string
			change func(*call)
		}{
			{"negative_row_stride", func(c *call) { c.rowStride = -8 }},
			{"negative_group_stride", func(c *call) { c.groupStride = -8 }},
			{"unaligned_row_stride", func(c *call) { c.rowStride -= 4 }},
			{"unaligned_group_stride", func(c *call) { c.groupStride -= 4 }},
			{"x_too_small", func(c *call) { c.x = Zeros(DTypeBFloat16, rows-1, 3, width) }},
			{"ends_out_of_order", func(c *call) { c.ends = [2]int{2100, 1000} }},
			{"ends_past_weight", func(c *call) { c.ends[1] = columns + 1 }},
			{"squares_width", func(c *call) { c.squares = Zeros(DTypeFloat32, 3, rows, width/128-1) }},
			{"squares_rows", func(c *call) { c.squares = Zeros(DTypeFloat32, 3, rows+1, width/128) }},
		} {
			c := valid
			tc.change(&c)
			if run(c) != nil {
				t.Errorf("%s: accepted", tc.name)
			}
		}
	})
}
