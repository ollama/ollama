// Package expertfixture builds routed-expert tensors in the on-disk layouts
// published MLX models ship, presented the way the runner hands them to model
// loaders. It exists for tests only.
package expertfixture

import (
	"fmt"
	"slices"
	"strings"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
)

// Encoding is how a fixture weight is stored.
type Encoding string

const (
	BF16  Encoding = "bf16"
	F32   Encoding = "f32"
	Int4  Encoding = "int4"
	MXFP8 Encoding = "mxfp8"
	NVFP4 Encoding = "nvfp4"
)

// Set collects tensors under their on-disk names together with the per-tensor
// quantization metadata the runtime reads from blob headers.
type Set struct {
	tensors map[string]*mlx.Array
	quant   map[string]*model.TensorQuantInfo
	seed    int
}

func New() *Set {
	return &Set{
		tensors: make(map[string]*mlx.Array),
		quant:   make(map[string]*model.TensorQuantInfo),
	}
}

// Raw stores a tensor unchanged.
func (s *Set) Raw(name string, a *mlx.Array) *Set {
	s.tensors[name] = a
	return s
}

// Weight stores a deterministic weight of the given shape at name, encoded as
// enc: quantized encodings also write name.scale and, for affine, name.bias.
func (s *Set) Weight(name string, enc Encoding, shape ...int) *Set {
	s.seed++
	n := 1
	for _, d := range shape {
		n *= d
	}
	values := make([]float32, n)
	for i := range values {
		values[i] = float32((i*7919+s.seed*104729)%2001-1000) / 8000
	}
	w := mlx.FromValues(values, shape...)
	switch enc {
	case F32:
		s.tensors[name] = w
		return s
	case BF16:
		s.tensors[name] = w.AsType(mlx.DTypeBFloat16)
		return s
	}
	groupSize, bits, mode := model.QuantizationParams(string(enc))
	q, scales, biases := mlx.Quantize(w.AsType(mlx.DTypeBFloat16), groupSize, bits, mode)
	s.tensors[name] = q
	s.tensors[name+".scale"] = scales
	if biases != nil {
		s.tensors[name+".bias"] = biases
	}
	s.quant[name] = &model.TensorQuantInfo{QuantType: strings.ToUpper(string(enc)), GroupSize: groupSize}
	return s
}

// GlobalScale stores name.global_scale as an F32 vector, the per-expert bank
// shape stacked imports write.
func (s *Set) GlobalScale(name string, values ...float32) *Set {
	s.tensors[name+".global_scale"] = mlx.FromValues(values, len(values))
	return s
}

// ScalarGlobalScale stores name.global_scale with shape [], as per-expert and
// per-tensor ModelOpt exports ship it.
func (s *Set) ScalarGlobalScale(name string, value float32) *Set {
	s.tensors[name+".global_scale"] = mlx.FromValue(value)
	return s
}

// PerExpert stores one [out, in] weight per expert at
// <group>.<expert>.<proj>.weight. A non-nil globalScales gives each expert a
// scalar global scale.
func (s *Set) PerExpert(group, proj string, enc Encoding, experts, out, in int, globalScales []float32) *Set {
	for e := range experts {
		name := fmt.Sprintf("%s.%d.%s.weight", group, e, proj)
		s.Weight(name, enc, out, in)
		if globalScales != nil {
			s.ScalarGlobalScale(name, globalScales[e])
		}
	}
	return s
}

// Delete removes on-disk tensors and their metadata.
func (s *Set) Delete(names ...string) *Set {
	for _, name := range names {
		delete(s.tensors, name)
		delete(s.quant, name)
	}
	return s
}

// Tensors returns the tensors renamed as the runner presents them to
// LoadWeights: X.scale becomes X_scale, and X.bias becomes X_qbias when X.scale
// exists.
func (s *Set) Tensors() map[string]*mlx.Array {
	scaled := make(map[string]bool)
	out := make(map[string]*mlx.Array, len(s.tensors))
	for name, a := range s.tensors {
		if base, ok := strings.CutSuffix(name, ".scale"); ok {
			out[base+"_scale"] = a
			scaled[base] = true
		}
	}
	for name, a := range s.tensors {
		if strings.HasSuffix(name, ".scale") {
			continue
		}
		if base, ok := strings.CutSuffix(name, ".bias"); ok && scaled[base] {
			out[base+"_qbias"] = a
			continue
		}
		out[name] = a
	}
	return out
}

// Quant returns the per-tensor quantization metadata keyed by tensor name.
func (s *Set) Quant() map[string]*model.TensorQuantInfo {
	out := make(map[string]*model.TensorQuantInfo, len(s.quant))
	for name, info := range s.quant {
		copied := *info
		out[name] = &copied
	}
	return out
}

// Defaults returns the model-level quantization parameters a manifest whose
// first quantized blob is enc would resolve to.
func Defaults(enc Encoding) (groupSize, bits int, mode string) {
	if enc == BF16 || enc == F32 {
		return model.QuantizationParams("")
	}
	return model.QuantizationParams(string(enc))
}

// Bank is the observable state of one loaded expert projection.
type Bank struct {
	Weight, Scales, Biases, GlobalScale *mlx.Array
	GroupSize, Bits                     int
	Mode                                string
}

// String renders a bank as "dense BF16[4 64 128]" or
// "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4] gs=F32[4]".
func (b Bank) String() string {
	if b.Weight == nil {
		return "-"
	}
	if b.Scales == nil {
		return "dense " + Shape(b.Weight)
	}
	out := fmt.Sprintf("%s/%d/%d %s s=%s", b.Mode, b.GroupSize, b.Bits, Shape(b.Weight), Shape(b.Scales))
	if b.Biases != nil {
		out += " qb=" + Shape(b.Biases)
	}
	if b.GlobalScale != nil {
		out += " gs=" + Shape(b.GlobalScale)
	}
	return out
}

// Compare lists the differences between loaded and expected bank
// descriptions, one line per key, in key order.
func Compare(got, want map[string]string) []string {
	keys := make(map[string]bool, len(got)+len(want))
	for k := range got {
		keys[k] = true
	}
	for k := range want {
		keys[k] = true
	}
	sorted := make([]string, 0, len(keys))
	for k := range keys {
		sorted = append(sorted, k)
	}
	slices.Sort(sorted)
	var diffs []string
	for _, k := range sorted {
		g, gok := got[k]
		w, wok := want[k]
		if !gok {
			g = "(absent)"
		}
		if !wok {
			w = "(absent)"
		}
		if g != w {
			diffs = append(diffs, fmt.Sprintf("%s: got %q, want %q", k, g, w))
		}
	}
	return diffs
}

// Shape renders an array's dtype and dimensions, or "-" for nil.
func Shape(a *mlx.Array) string {
	if a == nil {
		return "-"
	}
	return fmt.Sprintf("%s%v", a.DType(), a.Dims())
}
