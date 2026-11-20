package create

import (
	"encoding/binary"
	"io"
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

// e4m3Values are exact in E4M3, so encoding and decoding them is lossless.
var e4m3Values = []float32{1, 2, 0.5, -1.5, 3, -4, 0.25, 6, -0.75, 1.25, 8, -2, 0.125, 1.75, -6, 448}

func TestDecodeSourceFP8Tensor(t *testing.T) {
	values := e4m3Values
	tests := []struct {
		name  string
		scale []float32
		shape []int
		want  func(r, c int) float32
	}{
		{"per tensor", []float32{0.5}, []int{}, func(r, c int) float32 { return 0.5 }},
		{"per tensor [1]", []float32{0.5}, []int{1}, func(r, c int) float32 { return 0.5 }},
	}
	mlxtest.Run(t, func(mt *mlxtest.T) {
		for _, tt := range tests {
			weight := mlx.ToFP8(mlx.FromValues(values, 4, 4))
			var scale *mlx.Array
			if len(tt.shape) == 0 {
				scale = mlx.FromValue(tt.scale[0])
			} else {
				scale = mlx.FromValues(tt.scale, tt.shape...)
			}
			got, err := decodeSourceFP8Tensor(weight, scale)
			if err != nil {
				t.Errorf("%s: %v", tt.name, err)
				continue
			}
			out := got.AsType(mlx.DTypeFloat32)
			mlx.Eval(out)
			want := make([]float32, len(values))
			for i, v := range values {
				want[i] = v * tt.want(i/4, i%4)
			}
			if got.DType() != mlx.DTypeBFloat16 || !slices.Equal(out.Floats(), want) {
				t.Errorf("%s: decoded %v %v, want BF16 %v", tt.name, got.DType(), out.Floats(), want)
			}
		}

		// A per-expert bank: one scale per expert.
		bank := mlx.ToFP8(mlx.FromValues(append(slices.Clone(values), values...), 2, 4, 4))
		for _, scale := range []*mlx.Array{mlx.FromValues([]float32{1, 2}, 2), mlx.FromValues([]float32{1, 2}, 2, 1, 1)} {
			got, err := decodeSourceFP8Tensor(bank, scale)
			if err != nil {
				t.Fatal(err)
			}
			out := got.AsType(mlx.DTypeFloat32)
			mlx.Eval(out)
			f := out.Floats()
			if f[0] != values[0] || f[16] != 2*values[0] || f[31] != 2*values[15] {
				t.Errorf("per-expert scale %v: decoded %v", scale.Dims(), f)
			}
		}

		if _, err := decodeSourceFP8Tensor(mlx.ToFP8(mlx.FromValues(values, 4, 4)), mlx.FromValues([]float32{1, 2, 3, 4}, 4)); err == nil {
			t.Error("decode accepted a per-row scale")
		}
	})
}

// TestExactMXFP8MatchesFP8 checks that the stored mxfp8 weight, block scales
// and global scale reproduce the FP8 checkpoint's values bit for bit through
// MLX's own mxfp8 dequantize.
func TestExactMXFP8MatchesFP8(t *testing.T) {
	const rows, cols = 4, 64
	values := make([]float32, rows*cols)
	for i := range values {
		values[i] = e4m3Values[(i*7)%len(e4m3Values)]
	}
	tests := []struct {
		name   string
		scales []float32
		shape  []int32
	}{
		{"small", []float32{0.0013427734}, []int32{}},
		{"above one", []float32{3.7}, []int32{1}},
		{"power of two", []float32{1.0 / 448}, []int32{}},
		{"zero", []float32{0}, []int32{}},
	}
	mlxtest.Run(t, func(mt *mlxtest.T) {
		codes := mlx.ToFP8(mlx.FromValues(values, rows, cols)).AsType(mlx.DTypeInt32)
		mlx.Eval(codes)
		fp8 := make([]byte, rows*cols)
		for i, c := range codes.Ints() {
			fp8[i] = byte(c)
		}
		for _, tt := range tests {
			weight := SourceTensor{Name: "w.weight", Dtype: "F8_E4M3", Shape: []int32{rows, cols}}
			scale := SourceTensor{Name: "w.weight_scale", Dtype: "F32", Shape: tt.shape}
			spec, ok := exactMXFP8(weight, fp8Weight{scale: scale})
			if !ok {
				t.Fatalf("%s: weight did not convert exactly", tt.name)
			}
			scaleBytes := make([]byte, 4*len(tt.scales))
			for i, s := range tt.scales {
				binary.LittleEndian.PutUint32(scaleBytes[4*i:], math.Float32bits(s))
			}
			// The writer resolves a fresh reader for every tensor it writes.
			source := func(name string) *safetensors.TensorData {
				if name == weight.Name {
					return safetensors.NewTensorDataFromBytes(weight.Name, weight.Dtype, weight.Shape, fp8)
				}
				return safetensors.NewTensorDataFromBytes(scale.Name, scale.Dtype, scale.Shape, scaleBytes)
			}
			stored := make(map[string][]byte)
			for _, ts := range spec.Tensors {
				td, err := applyByteTransform(ts, []*safetensors.TensorData{source(ts.Sources[0].Name)})
				if err != nil {
					t.Fatalf("%s %s: %v", tt.name, ts.Name, err)
				}
				b, err := io.ReadAll(td.Reader())
				if err != nil {
					t.Fatal(err)
				}
				stored[ts.Name] = b
			}

			words := make([]uint32, rows*cols/4)
			for i := range words {
				words[i] = binary.LittleEndian.Uint32(stored["w.weight"][4*i:])
			}
			globals := make([]float32, len(stored["w.weight.global_scale"])/4)
			for i := range globals {
				globals[i] = math.Float32frombits(binary.LittleEndian.Uint32(stored["w.weight.global_scale"][4*i:]))
			}
			global := mlx.FromValues(globals, len(globals))
			blockScales := mlx.FromValues(stored["w.weight.scale"], rows, cols/mxfp8GroupSize)
			dequant := mlx.Dequantize(mlx.FromValues(words, rows, cols/4), blockScales, nil, mxfp8GroupSize, 8, "mxfp8", nil)
			got := mlx.Mul(dequant.AsType(mlx.DTypeFloat32), global)
			mlx.Eval(got)
			for i, g := range got.Floats() {
				want := values[i] * tt.scales[0]
				if math.Float32bits(g) != math.Float32bits(want) {
					t.Fatalf("%s: element %d = %v, want %v", tt.name, i, g, want)
				}
			}
		}
	})
}
