package create

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/x448/float16"

	"github.com/ollama/ollama/fs/safetensors"
)

// fp8DecodeTarget is what an FP8 weight with a float scale becomes: mxfp8
// holding exactly the published values. The FP8 bytes are mxfp8 elements, the
// scale's power of two goes into the block scales and the rest into the global
// scale. A weight the policy keeps at source precision is widened to BF16; an
// FP8 weight is never re-quantized.
const fp8DecodeTarget = "mxfp8"

// fp8Weight is an F8_E4M3 weight with one float scale for the tensor (per
// expert for a stacked bank).
type fp8Weight struct {
	scale      SourceTensor
	companions []string // source tensors folded into the weight's blob
}

// fp8ActivationSuffixes are activation scales that ship beside an FP8 weight;
// weight-only inference does not use them.
var fp8ActivationSuffixes = []string{".input_scale", ".input_global_scale"}

// matchFP8 reports whether name is an FP8 weight with a float scale. Block
// scales (DeepSeek-style 128x128) are an error: mxfp8 holds one power-of-two
// scale per 32 inputs, so storing them would re-round every weight.
func matchFP8(name string, inv Inventory) (fp8Weight, bool, error) {
	t, ok := inv.Tensors[name]
	if !ok || !strings.HasSuffix(name, ".weight") || !isE4M3Dtype(t.Dtype) || len(t.Shape) < 2 {
		return fp8Weight{}, false, nil
	}
	var scale SourceTensor
	for _, suffix := range []string{"_scale_inv", "_scale"} {
		if s, ok := inv.Tensors[name+suffix]; ok && isFloatScaleDtype(s.Dtype) {
			scale = s
			break
		}
	}
	if scale.Name == "" {
		return fp8Weight{}, false, nil
	}

	w := fp8Weight{scale: scale, companions: []string{scale.Name}}
	base := strings.TrimSuffix(name, ".weight")
	for _, suffix := range fp8ActivationSuffixes {
		if inv.Has(base + suffix) {
			w.companions = append(w.companions, base+suffix)
		}
	}

	rowShape := t.Shape[:len(t.Shape)-1]
	switch {
	case elementCount(scale.Shape) == elementCount(rowShape[:len(rowShape)-1]):
		// One scale per tensor (per expert for a stacked bank).
	case elementCount(scale.Shape) == elementCount(rowShape) && len(scale.Shape) >= len(rowShape) && slices.Equal(scale.Shape[:len(rowShape)], rowShape):
		// Per-channel scales would import exactly, but applying per-row global scales is not yet
		// optimized (a 5-20% generation hit, worst on MoE experts), so this import path is rejected.
		return fp8Weight{}, false, fmt.Errorf("fp8 weight %s %v has per-channel scales %s %v: per-channel FP8 import is not supported yet", name, t.Shape, scale.Name, scale.Shape)
	default:
		return fp8Weight{}, false, fmt.Errorf("fp8 weight %s %v has block scales %s %v: block-scaled FP8 cannot be stored without re-rounding every weight", name, t.Shape, scale.Name, scale.Shape)
	}
	return w, true, nil
}

// mxfp8GroupSize is the number of elements one mxfp8 block scale covers.
const mxfp8GroupSize = 32

// exactMXFP8 returns weight as an mxfp8 blob holding exactly the values its
// FP8 bytes and scale encode, or false when the input dimension does not
// divide into mxfp8 blocks.
func exactMXFP8(weight SourceTensor, w fp8Weight) (BlobSpec, bool) {
	rank := len(weight.Shape)
	if weight.Shape[rank-1]%mxfp8GroupSize != 0 {
		return BlobSpec{}, false
	}
	rowShape := weight.Shape[:rank-1]
	cols := weight.Shape[rank-1]
	// One global scale per tensor, per expert for a stacked bank.
	global := slices.Clone(rowShape[:len(rowShape)-1])
	return BlobSpec{
		Name: weight.Name,
		Tensors: []TensorSpec{
			{Name: weight.Name, Sources: []SourceTensor{weight}, Transform: TransformRepackFP8, OutDtype: "U32", OutShape: append(slices.Clone(rowShape), cols/4)},
			{Name: weight.Name + ".scale", Sources: []SourceTensor{w.scale}, Transform: TransformFP8ScaleExponent, OutDtype: "U8", OutShape: append(slices.Clone(rowShape), cols/mxfp8GroupSize)},
			{Name: weight.Name + ".global_scale", Sources: []SourceTensor{w.scale}, Transform: TransformFP8ScaleRemainder, OutDtype: "F32", OutShape: global},
		},
		Metadata: map[string]string{"quant_type": "mxfp8", "group_size": strconv.Itoa(mxfp8GroupSize)},
	}, true
}

// planFP8Weight stores an FP8 weight exactly as mxfp8, or widens it to BF16
// when the policy keeps it at source precision or it cannot convert exactly.
// It never re-quantizes.
func planFP8Weight(t SourceTensor, w fp8Weight, quantize string) (BlobSpec, error) {
	switch quantize {
	case fp8DecodeTarget:
		if spec, ok := exactMXFP8(t, w); ok {
			return spec, nil
		}
	case "":
	default:
		return BlobSpec{}, fmt.Errorf("fp8 weight %s would be re-quantized as %s", t.Name, quantize)
	}
	return BlobSpec{Name: t.Name, Tensors: []TensorSpec{{
		Name:      t.Name,
		Sources:   []SourceTensor{t, w.scale},
		Transform: TransformDecodeFP8,
		OutDtype:  "BF16",
		OutShape:  t.Shape,
	}}}, nil
}

// exactFP8ExpertGroup returns the exact mxfp8 blob of every weight in an
// expert group, keyed by source name, or false when any weight cannot convert
// exactly or the policy wants a projection as anything but mxfp8.
func exactFP8ExpertGroup(group string, weights []SourceTensor, fp8 map[string]fp8Weight, target string, policy quantizePolicy) (map[string]BlobSpec, bool) {
	exact := make(map[string]BlobSpec, len(weights))
	experts := make(map[string]int)
	shapes := make(map[string][]int32)
	for _, w := range weights {
		spec, ok := exactMXFP8(w, fp8[w.Name])
		if !ok {
			return nil, false
		}
		_, proj, err := parseExpertTensor(group, w.Name)
		if err != nil {
			return nil, false
		}
		exact[w.Name] = spec
		experts[proj]++
		shapes[proj] = w.Shape
	}
	for proj, n := range experts {
		name := group + "." + proj + ".weight"
		if policy.quantizationType(name, append([]int32{int32(n)}, shapes[proj]...), target) != "mxfp8" {
			return nil, false
		}
	}
	return exact, true
}

// splitFP8Scales reads an FP8 scale tensor and splits every scale s into
// 2^exponent * remainder with remainder in [1, 2). A zero scale keeps a zero
// remainder.
func splitFP8Scales(td *safetensors.TensorData) (exponents []int, remainders []float32, err error) {
	raw, err := io.ReadAll(td.Reader())
	if err != nil {
		return nil, nil, err
	}
	var values []float32
	switch strings.ToUpper(td.Dtype) {
	case "F32":
		for i := 0; i+4 <= len(raw); i += 4 {
			values = append(values, math.Float32frombits(binary.LittleEndian.Uint32(raw[i:])))
		}
	case "BF16":
		for i := 0; i+2 <= len(raw); i += 2 {
			values = append(values, math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i:]))<<16))
		}
	case "F16":
		for i := 0; i+2 <= len(raw); i += 2 {
			values = append(values, float16.Frombits(binary.LittleEndian.Uint16(raw[i:])).Float32())
		}
	default:
		return nil, nil, fmt.Errorf("fp8 scale %s has dtype %s, want F32, BF16 or F16", td.Name, td.Dtype)
	}
	if int64(len(values)) != elementCount(td.Shape) {
		return nil, nil, fmt.Errorf("fp8 scale %s holds %d values for shape %v", td.Name, len(values), td.Shape)
	}
	exponents = make([]int, len(values))
	remainders = make([]float32, len(values))
	for i, v := range values {
		if v == 0 {
			continue
		}
		if !(v > 0) || math.IsInf(float64(v), 0) {
			return nil, nil, fmt.Errorf("fp8 scale %s has value %v, want a finite non-negative scale", td.Name, v)
		}
		frac, exp := math.Frexp(float64(v))
		exponents[i], remainders[i] = exp-1, float32(2*frac)
		if exp-1 < -127 || exp-1 > 127 {
			return nil, nil, fmt.Errorf("fp8 scale %s value %v is outside the E8M0 range", td.Name, v)
		}
	}
	return exponents, remainders, nil
}

// fp8ScaleExponents writes the [..., rows, blocks] E8M0 block scales of shape
// from one scale per [rows, cols] matrix.
func fp8ScaleExponents(td *safetensors.TensorData, name string, shape []int32) (*safetensors.TensorData, error) {
	if len(shape) < 2 {
		return nil, fmt.Errorf("fp8 block scales %s need a [..., rows, blocks] shape, got %v", name, shape)
	}
	exponents, _, err := splitFP8Scales(td)
	if err != nil {
		return nil, err
	}
	blocks := int(shape[len(shape)-1])
	rowsPerScale := int(shape[len(shape)-2])
	rows := int(elementCount(shape[:len(shape)-1]))
	if len(exponents)*rowsPerScale != rows {
		return nil, fmt.Errorf("fp8 scale %s has %d values for block scales %v", td.Name, len(exponents), shape)
	}
	out := make([]byte, rows*blocks)
	for r := range rows {
		e := byte(127 + exponents[r/rowsPerScale])
		for b := range blocks {
			out[r*blocks+b] = e
		}
	}
	return safetensors.NewTensorDataFromBytes(name, "U8", slices.Clone(shape), out), nil
}

// fp8ScaleRemainders writes the F32 global scale left after the block scales
// take each FP8 scale's power of two.
func fp8ScaleRemainders(td *safetensors.TensorData, name string, shape []int32) (*safetensors.TensorData, error) {
	_, remainders, err := splitFP8Scales(td)
	if err != nil {
		return nil, err
	}
	if int64(len(remainders)) != elementCount(shape) {
		return nil, fmt.Errorf("fp8 scale %s has %d values for global scale shape %v", td.Name, len(remainders), shape)
	}
	out := make([]byte, 4*len(remainders))
	for i, r := range remainders {
		binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(r))
	}
	return safetensors.NewTensorDataFromBytes(name, "F32", slices.Clone(shape), out), nil
}

func isFloatScaleDtype(dtype string) bool {
	switch strings.ToUpper(dtype) {
	case "F32", "BF16", "F16":
		return true
	default:
		return false
	}
}

// planFP8ExpertGroup widens a layer's per-expert FP8 weights to BF16, for an
// expert group that cannot convert exactly as a whole: the experts of each
// projection are stacked into [experts, out, in] with their scales and decoded
// on the MLX writer thread. A policy that wants any projection quantized is an
// error, since that would re-quantize FP8.
func planFP8ExpertGroup(groupPrefix string, tensors []SourceTensor, fp8 map[string]fp8Weight, target string, policy quantizePolicy) ([]BlobSpec, error) {
	type expert struct {
		weight SourceTensor
		fp8    fp8Weight
	}
	byProj := make(map[string]map[int]expert)
	for _, t := range tensors {
		idx, proj, err := parseExpertTensor(groupPrefix, t.Name)
		if err != nil {
			return nil, err
		}
		if byProj[proj] == nil {
			byProj[proj] = make(map[int]expert)
		}
		if _, ok := byProj[proj][idx]; ok {
			return nil, fmt.Errorf("expert group %s projection %s has duplicate expert index %d", groupPrefix, proj, idx)
		}
		byProj[proj][idx] = expert{weight: t, fp8: fp8[t.Name]}
	}

	expertCount, err := validateExpertProjections(groupPrefix, byProj)
	if err != nil {
		return nil, err
	}
	var tensorSpecs []TensorSpec
	for _, proj := range sortedKeys(byProj) {
		experts := byProj[proj]
		base := experts[0]
		// Sources are the N weights followed by the N scales, in expert order,
		// matching what TransformDecodeStackFP8 expects.
		sources := make([]SourceTensor, 0, 2*expertCount)
		scales := make([]SourceTensor, 0, expertCount)
		for i := range expertCount {
			e := experts[i]
			if e.weight.Dtype != base.weight.Dtype || !slices.Equal(e.weight.Shape, base.weight.Shape) {
				return nil, fmt.Errorf("fp8 expert group %s projection %s has mismatched weight layout (%s %v vs %s %v)",
					groupPrefix, proj, base.weight.Dtype, base.weight.Shape, e.weight.Dtype, e.weight.Shape)
			}
			if e.fp8.scale.Dtype != base.fp8.scale.Dtype || !slices.Equal(e.fp8.scale.Shape, base.fp8.scale.Shape) {
				return nil, fmt.Errorf("fp8 expert group %s projection %s has mismatched scale layout (%s %v vs %s %v)",
					groupPrefix, proj, base.fp8.scale.Dtype, base.fp8.scale.Shape, e.fp8.scale.Dtype, e.fp8.scale.Shape)
			}
			sources = append(sources, e.weight)
			scales = append(scales, e.fp8.scale)
		}
		sources = append(sources, scales...)

		stackedName := groupPrefix + "." + proj + ".weight"
		stackedShape := append([]int32{int32(expertCount)}, base.weight.Shape...)
		if q := policy.quantizationType(stackedName, stackedShape, target); q != "" && q != fp8DecodeTarget {
			return nil, fmt.Errorf("fp8 expert bank %s would be re-quantized as %s", stackedName, q)
		}
		tensorSpecs = append(tensorSpecs, TensorSpec{
			Name:      stackedName,
			Sources:   sources,
			Transform: TransformDecodeStackFP8,
			OutDtype:  base.weight.Dtype,
			OutShape:  stackedShape,
		})
	}
	return homogeneousExpertBlobs(groupPrefix, tensorSpecs), nil
}
