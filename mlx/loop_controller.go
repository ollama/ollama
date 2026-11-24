package mlx

import "math"

// Loop controllers gate a looped layer's input with a message from the
// previous pass. Controller i reads input row r (shared by all controllers
// when inputs is 1) and message first+i of row r:
//
//	compressed = Down_i · [rms(input), rms(message)]
//	gain, forget = tanh of the FP32 products of compressed with the gates
//	out = input * (1 - forget) + message * gain
//
// Inputs are [..., inputs, width] (or [..., width] when inputs is 1) and
// messages [..., messages, width], both FP32. The kernels read them in place
// through their strides: the residual mixer returns each query's mixture as
// one contiguous block, so messages arrive as a transposed view that a
// contiguous binding would copy for every launch. Fast loads need a
// contiguous last axis, leading axes that merge into one row axis and
// row-contiguous weights, gates, norms and compressed values; any other
// layout gives the same result from element reads.

// controllerRows bounds the rows one short controller launch keeps in
// registers. Larger row counts take several tiles.
const controllerRows = 8

// controllerViewHeader reads the row and slot strides of a [..., (slots,)
// width] input whose leading axes merge into one row axis, and checks the
// layouts the fast loads assume. Other layouts are read element by element
// through controller_offset.
const controllerViewHeader = `
// controller_offset is the element offset of row-major index i in an array
// of any layout.
inline int64_t controller_offset(size_t i, const constant int* shape, const constant int64_t* strides, int ndim) {
    int64_t at = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        at += int64_t(i % size_t(shape[d])) * strides[d];
        i /= size_t(shape[d]);
    }
    return at;
}
struct ControllerView { int64_t row, slot; };
inline ControllerView controller_view(const constant int64_t* strides, int ndim, bool slotted) {
    int last = ndim - (slotted ? 2 : 1);
    return {last > 0 ? strides[last - 1] : 0, slotted ? strides[last] : 0};
}
inline bool controller_merges(const constant int* shape, const constant int64_t* strides, int ndim, bool slotted) {
    int last = ndim - (slotted ? 2 : 1);
    if (strides[ndim - 1] != 1) return false;
    for (int i = 0; i + 1 < last; ++i) {
        if (strides[i] != strides[i + 1] * shape[i + 1]) return false;
    }
    return true;
}
inline bool controller_dense(const constant int* shape, const constant int64_t* strides, int ndim) {
    int64_t expected = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        if (shape[i] > 1 && strides[i] != expected) return false;
        expected *= shape[i];
    }
    return true;
}
`

// LoopControllerDown returns compressed [count, rows, rank] in FP32 for count
// controllers with BF16 weights [count, rank, 2*width]. The unweighted RMS
// scales apply to the FP32 dot products of the input and message halves, and
// each result rounds through BF16 like the projection it replaces.
// It returns nil when the Metal kernel cannot run.
func LoopControllerDown(input *Array, inputs int, message *Array, first, count int, weight *Array, epsilon float32) *Array {
	rows, width, ok := controllerShapes(input, inputs, message, first, count)
	columns := loopControllerDownConfig(kernelArchitecture())
	if !ok || columns == 0 || weight == nil || weight.NumDims() != 3 || weight.DType() != DTypeBFloat16 ||
		weight.Dim(0) != count || weight.Dim(2) != 2*width || weight.Dim(1)%columns != 0 {
		return nil
	}
	rank, tile := weight.Dim(1), min(rows, controllerRows)
	out, launched := loopControllerDown.applyMetal(gpuLaunch{
		inputs: []*Array{input, message, weight},
		ints: []gpuIntArg{
			{"Rows", tile},
			{"Total", rows},
			{"Width", width},
			{"Rank", rank},
			{"Inputs", inputs},
			{"MessageFirst", first},
			{"Columns", columns},
			{"EpsBits", int(math.Float32bits(epsilon))},
		},
		outputs:     []gpuOutputSpec{{"loop_controller_down", []int32{int32(count), int32(rows), int32(rank)}, DTypeFloat32}},
		grid:        [3]int{rank / columns * 256, count, (rows + tile - 1) / tile},
		threadGroup: [3]int{256, 1, 1},
	})
	if !launched {
		return nil
	}
	return out[0]
}

// loopControllerDownConfig returns the compressed values each threadgroup of
// LoopControllerDown computes, or 0 where the separate normalization and down
// products apply.
func loopControllerDownConfig(architecture string) int {
	generation, _ := appleGPU(architecture)
	if generation < 15 {
		// M2 can reject these register-heavy 256-thread groups when they are
		// evaluated; no fallback catches that.
		return 0
	} else if generation == 16 {
		// M4 Pro: the separate products made one- and two-request forwards
		// 1-3% faster than the fused controllers.
		return 0
	}
	// Four values beat one or two at 1-8 rows on M5 Max.
	return 4
}

// LoopControllerGate returns the gated inputs [count, rows, width] in FP32
// from compressed [count, rows, rank] and FP32 gates [count, rank, 2*width],
// gain columns then forget columns. It returns nil when the Metal kernel
// cannot run.
func LoopControllerGate(input *Array, inputs int, message *Array, first int, compressed, gates *Array) *Array {
	out, _ := loopControllerGateLaunch(input, inputs, message, first, compressed, gates, nil)
	return out
}

// LoopControllerGateNorm is LoopControllerGate followed by the scale of an RMS
// normalization with weights norm [count, width]: it returns out*norm rounded
// to BF16 and the normalization's partial sums of squares
// [count, rows, width/128] of the FP32 gated inputs, which GroupedShortMatmul
// completes. It returns nils when the Metal kernel cannot run.
func LoopControllerGateNorm(input *Array, inputs int, message *Array, first int, compressed, gates, norm *Array) (out, squares *Array) {
	if norm == nil {
		return nil, nil
	}
	return loopControllerGateLaunch(input, inputs, message, first, compressed, gates, norm)
}

func loopControllerGateLaunch(input *Array, inputs int, message *Array, first int, compressed, gates, norm *Array) (out, squares *Array) {
	if compressed == nil || gates == nil || compressed.NumDims() != 3 || gates.NumDims() != 3 ||
		compressed.DType() != DTypeFloat32 || gates.DType() != DTypeFloat32 {
		return nil, nil
	}
	count, rank := compressed.Dim(0), compressed.Dim(2)
	rows, width, ok := controllerShapes(input, inputs, message, first, count)
	if !ok || width%128 != 0 || compressed.Dim(1) != rows || rank > 256 ||
		gates.Dim(0) != count || gates.Dim(1) != rank || gates.Dim(2) != 2*width ||
		uint64(count)*uint64(rows)*uint64(width) > math.MaxInt32 {
		return nil, nil
	}
	scaled, perGroup, squaresShape, dtype := 0, min(rows, controllerRows), []int32{1}, DTypeFloat32
	if norm != nil {
		if norm.DType() != DTypeFloat32 || norm.Size() != count*width {
			return nil, nil
		}
		scaled, dtype = 1, DTypeBFloat16
		squaresShape = []int32{int32(count), int32(rows), int32(width / 128)}
	} else {
		// The unscaled kernel binds gates in the norm slot but never reads it.
		norm = gates
	}
	result, launched := loopControllerGate.applyMetal(gpuLaunch{
		inputs: []*Array{input, message, compressed, gates, norm},
		dtypes: []gpuDTypeArg{{"OutT", dtype}},
		ints: []gpuIntArg{
			{"Width", width},
			{"Rank", rank},
			{"Inputs", inputs},
			{"MessageFirst", first},
			{"RowsPerGroup", perGroup},
			{"Scaled", scaled},
		},
		outputs: []gpuOutputSpec{
			{"loop_controller_gate", []int32{int32(count), int32(rows), int32(width)}, dtype},
			{"loop_controller_squares", squaresShape, DTypeFloat32},
		},
		grid:        [3]int{width, count, (rows + perGroup - 1) / perGroup},
		threadGroup: [3]int{128, 1, 1},
	})
	if !launched {
		return nil, nil
	}
	if scaled == 0 {
		return result[0], nil
	}
	return result[0], result[1]
}

// controllerShapes validates the shared input and message layout.
func controllerShapes(input *Array, inputs int, message *Array, first, count int) (rows, width int, ok bool) {
	if !MetalIsAvailable() || input == nil || message == nil || input.DType() != DTypeFloat32 ||
		message.DType() != DTypeFloat32 || message.NumDims() < 2 || count < 1 || (inputs != 1 && inputs != count) {
		return 0, 0, false
	}
	width, messages := message.Dim(-1), message.Dim(-2)
	if width == 0 || width%8 != 0 || input.Dim(-1) != width || first < 0 || first+count > messages {
		return 0, 0, false
	}
	rows = message.Size() / (messages * width)
	if rows < 1 || input.Size() != rows*inputs*width || (inputs > 1 && input.Dim(-2) != inputs) ||
		uint64(message.Size()) > math.MaxInt32 {
		return 0, 0, false
	}
	return rows, width, true
}

// Each threadgroup computes Columns compressed values of one controller.
// SIMD groups below Half read the input half of the weight rows, the others
// the message half, each taking every Half-th block of 256 columns; their
// sums and the halves' sums of squares combine in group order.
var loopControllerDown = &gpuKernel{
	name:    "loop_controller_down",
	inputs:  []string{"input", "message", "weight"},
	outputs: []string{"compressed"},
	strided: true,
	metal: gpuSource{header: load8Header + controllerViewHeader, source: `
constexpr uint Chunks = (Width + 255) / 256;
constexpr uint Half = 4; // half of the eight SIMD groups in a 256-thread group
const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const uint ctrl = threadgroup_position_in_grid.y;
const uint first = threadgroup_position_in_grid.x * Columns;
const uint base = threadgroup_position_in_grid.z * Rows;
const bool second = sg >= Half;
const uint h = second ? sg - Half : sg;
const ControllerView in_view = controller_view(input_strides, input_ndim, Inputs > 1);
const ControllerView msg_view = controller_view(message_strides, message_ndim, true);
// load8 reads 16-byte vectors, so rows and slots must stay 16-byte aligned.
const bool fast = controller_merges(input_shape, input_strides, input_ndim, Inputs > 1) &&
    controller_merges(message_shape, message_strides, message_ndim, true) &&
    controller_dense(weight_shape, weight_strides, weight_ndim) &&
    in_view.row % 4 == 0 && in_view.slot % 4 == 0 && msg_view.row % 4 == 0 && msg_view.slot % 4 == 0 &&
    (ulong(input) & 15) == 0 && (ulong(message) & 15) == 0 && (ulong(weight) & 15) == 0;
threadgroup float dots[2 * Half][Columns][Rows];
threadgroup float sums[2 * Half][Rows];
if (fast) {
    const ControllerView view = second ? msg_view : in_view;
    const device float* src = second ? message + (MessageFirst + ctrl) * view.slot
                                     : input + (Inputs == 1 ? 0u : ctrl) * view.slot;
    const device bfloat16_t* w = weight + (size_t(ctrl) * Rank + first) * (2 * Width) + (second ? Width : 0);
    float acc[Columns][Rows] = {};
    float squares[Rows] = {};
    for (uint chunk = h; chunk < Chunks; chunk += Half) {
        const uint k = chunk * 256 + lane * 8;
        if (k >= Width) continue;
        float wv[Columns][8];
        for (uint c = 0; c < Columns; ++c) load8(w + c * (2 * Width) + k, wv[c]);
        for (uint m = 0; m < Rows && base + m < Total; ++m) {
            float xv[8];
            load8(src + (base + m) * view.row + k, xv);
            for (uint j = 0; j < 8; ++j) {
                squares[m] = fma(xv[j], xv[j], squares[m]);
                for (uint c = 0; c < Columns; ++c) acc[c][m] = fma(xv[j], wv[c][j], acc[c][m]);
            }
        }
    }
    for (uint m = 0; m < Rows; ++m) {
        for (uint c = 0; c < Columns; ++c) {
            float s = simd_sum(acc[c][m]);
            if (lane == 0) dots[sg][c][m] = s;
        }
        float s = simd_sum(squares[m]);
        if (lane == 0) sums[sg][m] = s;
    }
} else {
    // Any other layout: the same sums in the same order from element reads,
    // one row at a time so this path holds fewer registers than the fast one.
    const device float* data = second ? message : input;
    const constant int* shape = second ? message_shape : input_shape;
    const constant int64_t* strides = second ? message_strides : input_strides;
    const int ndim = second ? message_ndim : input_ndim;
    const uint slots = second ? uint(message_shape[message_ndim - 2]) : Inputs;
    const uint slot = second ? MessageFirst + ctrl : (Inputs == 1 ? 0u : ctrl);
    const size_t w = (size_t(ctrl) * Rank + first) * (2 * Width) + (second ? Width : 0);
    for (uint m = 0; m < Rows; ++m) {
        float acc[Columns] = {};
        float squares = 0.0f;
        for (uint chunk = h; chunk < Chunks && base + m < Total; chunk += Half) {
            const uint k = chunk * 256 + lane * 8;
            if (k >= Width) continue;
            const size_t at = (size_t(base + m) * slots + slot) * Width + k;
            for (uint j = 0; j < 8; ++j) {
                const float x = data[controller_offset(at + j, shape, strides, ndim)];
                squares = fma(x, x, squares);
                for (uint c = 0; c < Columns; ++c) {
                    const float wj = float(weight[controller_offset(w + c * (2 * Width) + k + j, weight_shape, weight_strides, weight_ndim)]);
                    acc[c] = fma(x, wj, acc[c]);
                }
            }
        }
        for (uint c = 0; c < Columns; ++c) {
            float s = simd_sum(acc[c]);
            if (lane == 0) dots[sg][c][m] = s;
        }
        float s = simd_sum(squares);
        if (lane == 0) sums[sg][m] = s;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint t = thread_index_in_threadgroup;
if (t >= Columns * Rows) return;
const uint c = t / Rows, m = t % Rows;
if (base + m >= Total) return;
float a = 0.0f, b = 0.0f, sa = 0.0f, sb = 0.0f;
for (uint j = 0; j < Half; ++j) {
    a += dots[j][c][m];
    b += dots[Half + j][c][m];
    sa += sums[j][m];
    sb += sums[Half + j][m];
}
const float eps = as_type<float>(EpsBits);
float v = a * metal::precise::rsqrt(sa / float(Width) + eps) + b * metal::precise::rsqrt(sb / float(Width) + eps);
compressed[(size_t(ctrl) * Total + base + m) * Rank + first + c] = float(bfloat(v));
`},
}

// One thread owns a column of one controller for RowsPerGroup rows. The gate
// products accumulate over the rank in order, as the graph's FP32 product.
var loopControllerGate = &gpuKernel{
	name:    "loop_controller_gate",
	inputs:  []string{"input", "message", "compressed", "gates", "norm"},
	outputs: []string{"out", "squares"},
	strided: true,
	metal: gpuSource{header: controllerViewHeader, source: `
const uint rows = compressed_shape[1];
const uint c = thread_position_in_grid.x, ctrl = threadgroup_position_in_grid.y;
const uint base = threadgroup_position_in_grid.z * RowsPerGroup;
const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const ControllerView in_view = controller_view(input_strides, input_ndim, Inputs > 1);
const ControllerView msg_view = controller_view(message_strides, message_ndim, true);
const bool fast = controller_merges(input_shape, input_strides, input_ndim, Inputs > 1) &&
    controller_merges(message_shape, message_strides, message_ndim, true) &&
    controller_dense(compressed_shape, compressed_strides, compressed_ndim) &&
    controller_dense(gates_shape, gates_strides, gates_ndim) && controller_dense(norm_shape, norm_strides, norm_ndim);
threadgroup float partial[Scaled ? 4 * RowsPerGroup : 1];
if (!fast) {
    // Any other layout: the same values in the same order from element reads.
    const uint slot = Inputs == 1 ? 0u : ctrl;
    const uint messages = message_shape[message_ndim - 2];
    for (uint r = 0; r < RowsPerGroup; ++r) {
        const uint row = base + r;
        float y = 0.0f;
        if (row < rows) {
            float gain = 0.0f, forget = 0.0f;
            for (uint k = 0; k < Rank; ++k) {
                const float x = compressed[controller_offset((size_t(ctrl) * rows + row) * Rank + k, compressed_shape, compressed_strides, compressed_ndim)];
                const size_t at = (size_t(ctrl) * Rank + k) * (2 * Width) + c;
                gain = fma(x, gates[controller_offset(at, gates_shape, gates_strides, gates_ndim)], gain);
                forget = fma(x, gates[controller_offset(at + Width, gates_shape, gates_strides, gates_ndim)], forget);
            }
            float in = input[controller_offset((size_t(row) * Inputs + slot) * Width + c, input_shape, input_strides, input_ndim)];
            float msg = message[controller_offset((size_t(row) * messages + MessageFirst + ctrl) * Width + c, message_shape, message_strides, message_ndim)];
            float keep = 1.0f - metal::precise::tanh(forget);
            y = in * keep + msg * metal::precise::tanh(gain);
            float scale = Scaled ? float(norm[controller_offset(ctrl * Width + c, norm_shape, norm_strides, norm_ndim)]) : 1.0f;
            out[(size_t(ctrl) * rows + row) * Width + c] = OutT(Scaled ? y * scale : y);
        }
        if (Scaled) {
            float s = simd_sum(y * y);
            if (lane == 0) partial[sg * RowsPerGroup + r] = s;
        }
    }
    if (Scaled) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint r = thread_index_in_threadgroup;
        if (r < RowsPerGroup && base + r < rows) {
            float s = partial[r];
            for (uint j = 1; j < 4; ++j) s += partial[j * RowsPerGroup + r];
            squares[(size_t(ctrl) * rows + base + r) * (Width / 128) + threadgroup_position_in_grid.x] = s;
        }
    }
    return;
}
threadgroup float shared[RowsPerGroup * Rank];
for (uint i = thread_index_in_threadgroup; i < RowsPerGroup * Rank; i += threads_per_threadgroup.x) {
    uint r = base + i / Rank;
    shared[i] = r < rows ? compressed[(size_t(ctrl) * rows + r) * Rank + i % Rank] : 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const device float* g = gates + size_t(ctrl) * Rank * (2 * Width) + c;
float gain[RowsPerGroup] = {}, forget[RowsPerGroup] = {};
for (uint k = 0; k < Rank; ++k) {
    float a = g[k * (2 * Width)], b = g[k * (2 * Width) + Width];
    for (uint r = 0; r < RowsPerGroup; ++r) {
        gain[r] = fma(shared[r * Rank + k], a, gain[r]);
        forget[r] = fma(shared[r * Rank + k], b, forget[r]);
    }
}
for (uint r = 0; r < RowsPerGroup; ++r) {
    const uint row = base + r;
    float y = 0.0f;
    if (row < rows) {
        float in = input[row * in_view.row + (Inputs == 1 ? 0u : ctrl) * in_view.slot + c];
        float msg = message[row * msg_view.row + (MessageFirst + ctrl) * msg_view.slot + c];
        float keep = 1.0f - metal::precise::tanh(forget[r]);
        y = in * keep + msg * metal::precise::tanh(gain[r]);
        size_t at = (size_t(ctrl) * rows + row) * Width + c;
        out[at] = OutT(Scaled ? y * float(norm[ctrl * Width + c]) : y);
    }
    if (Scaled) {
        float s = simd_sum(y * y);
        if (lane == 0) partial[sg * RowsPerGroup + r] = s;
    }
}
if (Scaled) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r = thread_index_in_threadgroup;
    if (r < RowsPerGroup && base + r < rows) {
        float s = partial[r];
        for (uint j = 1; j < 4; ++j) s += partial[j * RowsPerGroup + r];
        squares[(size_t(ctrl) * rows + base + r) * (Width / 128) + threadgroup_position_in_grid.x] = s;
    }
}
`},
}

// GroupedShortMatmul multiplies up to three groups of rows by consecutive
// column ranges of one BF16 weight [N, width]: columns below ends[0] read
// group 0, those below ends[1] group 1 and the rest group 2. Row m of group g
// starts at element g*groupStride + m*rowStride of x (FP32 or BF16). Each row
// reduces in ShortMatmul's order. With squares [3, rows, partials] from
// LoopControllerGateNorm, the sums are scaled by that row's inverse RMS.
// The result [rows, N] is BF16. It returns nil when the Metal kernel cannot run.
func GroupedShortMatmul(x *Array, rows, rowStride, groupStride int, ends [2]int, weight, squares *Array, epsilon float32) *Array {
	if !MetalIsAvailable() || x == nil || weight == nil || weight.NumDims() != 2 || weight.DType() != DTypeBFloat16 ||
		(x.DType() != DTypeFloat32 && x.DType() != DTypeBFloat16) || rows < 1 {
		return nil
	}
	if !groupedShortMatmulRuns(rows, kernelArchitecture()) {
		return nil
	}
	columns, width := weight.Dim(0), weight.Dim(1)
	if width%8 != 0 || rowStride < 0 || groupStride < 0 || rowStride%8 != 0 || groupStride%8 != 0 ||
		ends[0] < 0 || ends[0] > ends[1] || ends[1] > columns ||
		2*groupStride+(rows-1)*rowStride+width > x.Size() || uint64(x.Size()) > math.MaxUint32 ||
		uint64(weight.Size()) > math.MaxUint32 || uint64(rows)*uint64(columns) > math.MaxUint32 {
		return nil
	}
	scaled, partials := 0, 1
	if squares != nil {
		if squares.DType() != DTypeFloat32 || squares.NumDims() != 3 || squares.Dim(0) != 3 || squares.Dim(1) != rows ||
			squares.Dim(2) > 32 || squares.Dim(2)*128 != width {
			return nil
		}
		scaled, partials = 1, squares.Dim(2)
	} else {
		// The unscaled kernel binds weight in the squares slot but never reads it.
		squares = weight
	}
	const simdgroups = 8
	tile := min(rows, controllerRows)
	out, ok := groupedShortMatmul.applyMetal(gpuLaunch{
		inputs: []*Array{x, squares, weight},
		ints: []gpuIntArg{
			{"Rows", tile},
			{"Total", rows},
			{"Width", width},
			{"RowStride", rowStride},
			{"GroupStride", groupStride},
			{"End0", ends[0]},
			{"End1", ends[1]},
			{"Scaled", scaled},
			{"Partials", partials},
			{"EpsBits", int(math.Float32bits(epsilon))},
		},
		outputs:     []gpuOutputSpec{{"grouped_short_matmul", []int32{int32(rows), int32(columns)}, DTypeBFloat16}},
		grid:        [3]int{(columns + simdgroups - 1) / simdgroups * 32 * simdgroups, (rows + tile - 1) / tile, 1},
		threadGroup: [3]int{32 * simdgroups, 1, 1},
	})
	if !ok {
		return nil
	}
	return out[0]
}

// groupedShortMatmulRuns reports whether one grouped launch replaces the
// separate products for rows.
func groupedShortMatmulRuns(rows int, architecture string) bool {
	generation, _ := appleGPU(architecture)
	if generation == 16 && rows > controllerRows {
		// M4 Pro computes 8-row products below its bandwidth; there the
		// scaled epilogue cost 12% per tile, and two tiles made a two-request
		// n8 forward 1.2% slower than the separate products.
		return false
	}
	return true
}

var groupedShortMatmul = &gpuKernel{
	name:    "grouped_short_matmul",
	inputs:  []string{"x", "squares", "weight"},
	outputs: []string{"y"},
	metal: gpuSource{header: load8Header, source: `
const uint columns = weight_shape[0];
const uint lane = thread_index_in_simdgroup;
const uint col = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
if (col >= columns) return;
const uint base = threadgroup_position_in_grid.y * Rows;
const uint group = col < End0 ? 0 : col < End1 ? 1 : 2;
auto xs = x + group * GroupStride + base * RowStride;
const device bfloat16_t* w = weight + size_t(col) * Width;
float acc[Rows] = {};
for (uint k0 = 0; k0 < Width; k0 += 256) {
    const uint k = k0 + lane * 8;
    if (k >= Width) continue;
    float wv[8];
    load8(w + k, wv);
    for (uint m = 0; m < Rows && base + m < Total; ++m) {
        float xv[8];
        load8(xs + m * RowStride + k, xv);
        for (uint j = 0; j < 8; ++j) acc[m] = fma(xv[j], wv[j], acc[m]);
    }
}
for (uint m = 0; m < Rows && base + m < Total; ++m) {
    float sum = simd_sum(acc[m]);
    if (Scaled) {
        float s = simd_sum(lane < Partials ? float(squares[(group * Total + base + m) * Partials + lane]) : 0.0f);
        sum *= metal::precise::rsqrt(s / float(Width) + as_type<float>(EpsBits));
    }
    if (lane == 0) y[(base + m) * columns + col] = bfloat(sum);
}
`},
}
