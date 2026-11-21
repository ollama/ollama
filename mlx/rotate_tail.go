package mlx

const (
	rotateTailInput = iota
	rotateTailPositions
	rotateTailFrequencies
)

const (
	rotateTailWidth = iota
	rotateTailRotary
	rotateTailHeads
	rotateTailInverse
)

// One thread per output channel replaces the graph's slices, trig, products,
// stack and concatenation, with the graph's precise trig and FP32 rounding.
var rotateTail = &gpuKernel{
	name:    "rotate_tail",
	inputs:  []string{rotateTailInput: "x", rotateTailPositions: "positions", rotateTailFrequencies: "freqs"},
	outputs: []string{"out"},
	metal: gpuSource{header: `
#pragma METAL fp contract(off)
`, source: `
uint col = thread_position_in_grid.x % W;
uint row = thread_position_in_grid.x / W;
size_t at = size_t(row) * W + col;
int pass = W - R;
if (int(col) < pass) {
  out[at] = x[at];
  return;
}
int pair = (int(col) - pass) / 2;
size_t first = size_t(row) * W + pass + 2 * pair;
float re = float(x[first]);
float im = float(x[first + 1]);
float angle = float(positions[row / H]) * freqs[pair];
float c = metal::precise::cos(angle);
float s = metal::precise::sin(angle);
if (INV) {
  s = -s;
}
float v = ((int(col) - pass) % 2 == 0) ? re * c - im * s : re * s + im * c;
out[at] = static_cast<T>(v);
`},
	cuda: gpuSource{source: `
size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
if (index >= static_cast<size_t>(x_shape[0]) * x_shape[1] * x_shape[2] * W) {
  return;
}
int col = static_cast<int>(index % W);
int row = static_cast<int>(index / W);
size_t at = static_cast<size_t>(row) * W + col;
int pass = W - R;
if (col < pass) {
  out[at] = x[at];
  return;
}
int pair = (col - pass) / 2;
size_t first = static_cast<size_t>(row) * W + pass + 2 * pair;
float re = static_cast<float>(x[first]);
float im = static_cast<float>(x[first + 1]);
float angle = __fmul_rn(static_cast<float>(positions[row / H]), freqs[pair]);
float c = cosf(angle);
float s = sinf(angle);
if (INV) {
  s = -s;
}
float v = ((col - pass) % 2 == 0) ? __fsub_rn(__fmul_rn(re, c), __fmul_rn(im, s)) : __fadd_rn(__fmul_rn(re, s), __fmul_rn(im, c));
out[at] = static_cast<T>(v);
`},
	fallback: func(launch gpuLaunch) []*Array {
		in := launch.inputs
		return []*Array{rotateTailGraph(in[rotateTailInput], in[rotateTailPositions], in[rotateTailFrequencies], launch.ints[rotateTailInverse].value != 0)}
	},
}

func rotateTailGraph(x, positions, frequencies *Array, inverse bool) *Array {
	shape := x.Dims()
	batch, length, heads, width := int32(shape[0]), int32(shape[1]), int32(shape[2]), int32(shape[3])
	rotaryWidth := int32(frequencies.Size() * 2)
	pass := x.Slice(Slice(), Slice(), Slice(), Slice(0, int(width-rotaryWidth)))
	tail := x.Slice(Slice(), Slice(), Slice(), Slice(int(width-rotaryWidth), End))
	pairs := Reshape(tail.AsType(DTypeFloat32), batch, length, heads, rotaryWidth/2, 2)
	real := Squeeze(pairs.Slice(Slice(), Slice(), Slice(), Slice(), Slice(0)), -1)
	imag := Squeeze(pairs.Slice(Slice(), Slice(), Slice(), Slice(), Slice(1)), -1)
	angles := Mul(Reshape(positions.AsType(DTypeFloat32), batch, length, 1, 1), Reshape(frequencies, 1, 1, 1, rotaryWidth/2))
	cosine, sine := Cos(angles), Sin(angles)
	if inverse {
		sine = Neg(sine)
	}
	rotatedReal := Sub(Mul(real, cosine), Mul(imag, sine))
	rotatedImag := Add(Mul(real, sine), Mul(imag, cosine))
	rotated := Reshape(Stack([]*Array{rotatedReal, rotatedImag}, -1), batch, length, heads, rotaryWidth).AsType(x.DType())
	if rotaryWidth == width {
		return rotated
	}
	return Concatenate([]*Array{pass, rotated}, -1)
}

// RotateTail applies adjacent-pair rotary embeddings to the last
// 2*len(frequencies) channels of x [B, L, H, W] at positions [B, L], each
// pair turned by position*frequency, in FP32; inverse turns it back. The
// leading channels pass through.
func RotateTail(x, positions *Array, frequencies []float32, inverse bool) *Array {
	if x.NumDims() != 4 || positions.NumDims() != 2 || positions.Dim(0) != x.Dim(0) || positions.Dim(1) != x.Dim(1) ||
		len(frequencies) == 0 || 2*len(frequencies) > x.Dim(3) {
		panic("mlx: RotateTail requires x [B, L, H, W], positions [B, L], and 2*len(frequencies) <= W")
	}
	freqs := FromValues(frequencies, len(frequencies))
	rows := x.Dim(0) * x.Dim(1) * x.Dim(2)
	if rows == 0 {
		return rotateTailGraph(x, positions, freqs, inverse)
	}
	inv := 0
	if inverse {
		inv = 1
	}
	width := x.Dim(3)
	shape := make([]int32, 4)
	for i, d := range x.Dims() {
		shape[i] = int32(d)
	}
	return rotateTail.run(gpuLaunch{
		dtypes:  []gpuDTypeArg{{"T", x.DType()}},
		ints:    []gpuIntArg{rotateTailWidth: {"W", width}, rotateTailRotary: {"R", 2 * len(frequencies)}, rotateTailHeads: {"H", x.Dim(2)}, rotateTailInverse: {"INV", inv}},
		outputs: []gpuOutputSpec{{"ROTATE_TAIL", shape, x.DType()}},
		// One axis avoids CUDA's 65535 grid-height limit on long prompts.
		grid:        [3]int{width * rows, 1, 1},
		threadGroup: [3]int{min(width*rows, 256), 1, 1},
		inputs:      []*Array{rotateTailInput: x, rotateTailPositions: positions.AsType(DTypeInt32), rotateTailFrequencies: freqs},
	})[0]
}
