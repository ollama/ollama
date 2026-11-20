package create

import (
	"fmt"
	"maps"
	"slices"
	"sort"
	"strconv"
	"strings"
)

// Transform names how a tensor's source(s) are turned into the output tensor.
// The zero value, TransformNone, copies a single source through unchanged.
type Transform string

const (
	TransformNone Transform = ""

	// TransformRepackFP4 reinterprets a U8 fp4-packed weight (2 values/byte)
	// as U32 words (8 values/word): the bytes are unchanged, only the dtype
	// and last dimension are relabeled.
	TransformRepackFP4 Transform = "repack_fp4"

	// TransformRepackFP8 reinterprets four FP8 values as one U32 word for
	// MLX's mxfp8 matrix multiplication. The bytes are unchanged.
	TransformRepackFP8 Transform = "repack_fp8"

	// TransformRelabelU8 relabels an F8_E4M3 scale as U8 so the loader reads
	// its raw bytes; the bytes themselves are unchanged.
	TransformRelabelU8 Transform = "relabel_u8"

	// TransformScalarF32 validates that the source is a scalar F32 and copies
	// it through (a global scale stored as-is).
	TransformScalarF32 Transform = "scalar_f32"

	// TransformF32 preserves an F32 vector scale bank with one value per expert.
	TransformF32 Transform = "f32"

	// TransformReciprocalF32 validates an F32 scale tensor and stores each
	// element's reciprocal (global scales the producer stored inverted).
	TransformReciprocalF32 Transform = "reciprocal_f32"

	// TransformFP8ScaleExponent turns an FP8 weight's per-tensor float scale
	// into its mxfp8 block scales: every block holds the power of two of the
	// scale, as E8M0. OutShape is [rows, blocks].
	TransformFP8ScaleExponent Transform = "fp8_scale_exponent"

	// TransformFP8ScaleRemainder stores what TransformFP8ScaleExponent leaves
	// of each FP8 scale, a factor in [1, 2), as the F32 global scale.
	TransformFP8ScaleRemainder Transform = "fp8_scale_remainder"

	// TransformStackExperts concatenates N per-expert source tensors (in
	// expert-index order) into one [experts, ...] tensor, applying
	// TensorSpec.Each to every source first.
	TransformStackExperts Transform = "stack_experts"

	// TransformDecodeFP8 widens an FP8 weight to BF16 using its per-tensor
	// float scale. Its two sources are the F8_E4M3 weight and its scale
	// companion.
	TransformDecodeFP8 Transform = "decode_fp8"

	// TransformDecodeStackFP8 stacks N per-expert FP8 weights (and their N
	// scales) into one [experts, out, in] tensor and widens it to BF16. Its
	// sources are the N weights followed by the N scales, in expert-index order.
	TransformDecodeStackFP8 Transform = "decode_stack_fp8"
)

// TensorSpec describes one output tensor within a blob: the source tensor(s)
// it is built from, the transform that combines or converts them, the name it
// takes in the blob, and an optional quantization to apply. When Quantize is
// set the writer runs MLX quantization, which generates the tensor's scale and
// bias sub-tensors; otherwise the (transformed) bytes are stored as-is.
type TensorSpec struct {
	Name      string
	Sources   []SourceTensor
	Transform Transform
	Quantize  string
	OutDtype  string  // dtype after the transform; "" means same as the single source
	OutShape  []int32 // shape after the transform; nil means same as the single source

	Each Transform // TransformStackExperts: byte transform applied to each source before stacking
}

// BlobSpec describes one output blob: its layer name, the tensors it contains,
// and its safetensors metadata. The planner builds these purely from the
// inventory and classification; the writer executes them and makes no
// decisions of its own.
type BlobSpec struct {
	Name     string
	Tensors  []TensorSpec
	Metadata map[string]string
}

// quantizePolicy decides the quantization type for each tensor of a model,
// returning "" to keep it at source precision. A policy may return a higher-
// precision type than requested for sensitive tensors. The per-architecture
// import transforms implement it; defaultQuantPolicy provides the generic
// default (GetTensorQuantization).
type quantizePolicy interface {
	quantizationType(name string, shape []int32, requested string) string
}

// Plan turns an inventory and its classification into the ordered list of
// blobs to write. It reads no weight data and makes every decision here, so
// the writer that follows has nothing left to decide. The policy decides which
// weights are quantized and to what; pass defaultQuantPolicy{} for the generic
// policy.
func Plan(inv Inventory, class Classification, policy quantizePolicy) ([]BlobSpec, error) {
	var floatTarget, fp8Target string
	switch class.Kind {
	case SourceFloat:
		floatTarget = class.Quantize
	case SourceFP8:
		fp8Target = class.Quantize
	case SourcePrequantized:
		fp8Target = fp8DecodeTarget
	default:
		return nil, fmt.Errorf("plan: source kind %q is not yet supported", class.Kind)
	}
	specs, err := planSource(inv, floatTarget, fp8Target, policy)
	if err != nil {
		return nil, err
	}
	if err := checkOutputCollisions(specs); err != nil {
		return nil, err
	}
	return specs, nil
}

// checkOutputCollisions rejects a plan in which two source tensors normalized
// to the same output name — for example a source shipping both foo.weight and
// foo.weight_packed, which would both fuse to foo.weight. Writing such a plan
// would produce blobs that silently shadow each other at load time.
func checkOutputCollisions(specs []BlobSpec) error {
	blobs := make(map[string]bool, len(specs))
	tensors := make(map[string]string)
	for _, spec := range specs {
		if blobs[spec.Name] {
			return fmt.Errorf("plan: two blobs named %s (source tensors normalize to a clashing name)", spec.Name)
		}
		blobs[spec.Name] = true
		for _, ts := range spec.Tensors {
			if prev, ok := tensors[ts.Name]; ok {
				return fmt.Errorf("plan: output tensor %s planned in both blob %s and blob %s (source tensors normalize to a clashing name)", ts.Name, prev, spec.Name)
			}
			tensors[ts.Name] = spec.Name
		}
	}
	return nil
}

// planSource plans any source tensor by tensor. A weight already quantized in a
// form the runtime executes is fused with its companions and kept as
// published; an FP8 weight with a float scale is decoded and converted to
// fp8Target; every other tensor is float and takes floatTarget through the
// policy. Per-expert weights are stacked into one bank per projection whatever
// their encoding. Single tensors come first in name order, then the expert
// banks.
func planSource(inv Inventory, floatTarget, fp8Target string, policy quantizePolicy) ([]BlobSpec, error) {
	fused := make(map[string]BlobSpec)
	fp8 := make(map[string]fp8Weight)
	consumed := make(map[string]bool)
	for _, name := range sortedTensorNames(inv) {
		if spec, sources, ok := matchPrequant(name, inv); ok {
			fused[name] = spec
			for _, s := range sources {
				consumed[s] = true
			}
			continue
		}
		w, ok, err := matchFP8(name, inv)
		if err != nil {
			return nil, err
		}
		if ok {
			fp8[name] = w
			for _, s := range w.companions {
				consumed[s] = true
			}
		}
	}
	if err := applyMLXExportEmbeddingBits(inv, fused); err != nil {
		return nil, err
	}

	groups := make(map[string][]SourceTensor)
	var singles []string
	for _, name := range sortedTensorNames(inv) {
		if consumed[name] {
			continue
		}
		// A fused weight is grouped by its output name: compressed-tensors
		// ships it as ".weight_packed".
		out := name
		if spec, ok := fused[name]; ok {
			out = spec.Name
		}
		if gp, ok := perExpertGroup(out); ok {
			groups[gp] = append(groups[gp], inv.Tensors[name])
		} else {
			singles = append(singles, name)
		}
	}

	specs := make([]BlobSpec, 0, len(singles)+len(groups))
	for _, name := range singles {
		t := inv.Tensors[name]
		if spec, ok := fused[name]; ok {
			specs = append(specs, spec)
			continue
		}
		if w, ok := fp8[name]; ok {
			spec, err := planFP8Weight(t, w, policy.quantizationType(name, t.Shape, fp8Target))
			if err != nil {
				return nil, err
			}
			specs = append(specs, spec)
			continue
		}
		q := ""
		if floatTarget != "" {
			q = policy.quantizationType(name, t.Shape, floatTarget)
		}
		specs = append(specs, BlobSpec{
			Name:    name,
			Tensors: []TensorSpec{{Name: name, Sources: []SourceTensor{t}, Quantize: q}},
		})
	}

	for _, gp := range sortedKeys(groups) {
		members := groups[gp]
		var nFused, nFP8 int
		for _, t := range members {
			if _, ok := fused[t.Name]; ok {
				nFused++
			}
			if _, ok := fp8[t.Name]; ok {
				nFP8++
			}
		}
		var (
			groupSpecs []BlobSpec
			err        error
		)
		switch {
		case nFused == len(members):
			groupSpecs, err = planFusedExpertGroup(gp, members, fused)
		case nFP8 == len(members):
			if exact, ok := exactFP8ExpertGroup(gp, members, fp8, fp8Target, policy); ok {
				groupSpecs, err = planFusedExpertGroup(gp, members, exact)
			} else {
				groupSpecs, err = planFP8ExpertGroup(gp, members, fp8, fp8Target, policy)
			}
		case nFused == 0 && nFP8 == 0:
			groupSpecs, err = planExpertGroup(gp, members, floatTarget, policy)
		default:
			err = fmt.Errorf("expert group %s mixes encodings: %d prequantized, %d fp8, %d float of %d weights",
				gp, nFused, nFP8, len(members)-nFused-nFP8, len(members))
		}
		if err != nil {
			return nil, err
		}
		specs = append(specs, groupSpecs...)
	}
	return specs, nil
}

// planFusedExpertGroup stacks per-expert prequantized weights into one bank per
// projection. Every tensor of an expert's fused blob (weight, block scale,
// bias, global scale) is stacked with the same tensor of the other experts, so
// calibrated scales are carried over unchanged.
func planFusedExpertGroup(group string, weights []SourceTensor, fused map[string]BlobSpec) ([]BlobSpec, error) {
	// The float planner validates the expert indices and orders each
	// projection's experts; with no quantization it returns one blob. It parses
	// output names, so hand it the fused weights under those names.
	byOutput := make(map[string]BlobSpec, len(weights))
	outputs := make([]SourceTensor, len(weights))
	for i, w := range weights {
		spec := fused[w.Name]
		byOutput[spec.Name] = spec
		outputs[i] = w
		outputs[i].Name = spec.Name
	}
	specs, err := planExpertGroup(group, outputs, "", defaultQuantPolicy{})
	if err != nil {
		return nil, err
	}
	bank := &specs[0]
	projections := bank.Tensors
	bank.Tensors = nil
	bank.Metadata = nil
	for _, projection := range projections {
		count := len(projection.Sources)
		first := byOutput[projection.Sources[0].Name]
		for slot, ft := range first.Tensors {
			suffix := strings.TrimPrefix(ft.Name, first.Name)
			sources := make([]SourceTensor, count)
			for i, w := range projection.Sources {
				part := byOutput[w.Name]
				if len(part.Tensors) != len(first.Tensors) || !maps.Equal(part.Metadata, first.Metadata) {
					return nil, fmt.Errorf("expert group %s: %s does not match the layout of %s", group, w.Name, first.Name)
				}
				pt := part.Tensors[slot]
				if strings.TrimPrefix(pt.Name, part.Name) != suffix || len(pt.Sources) != 1 || pt.Transform != ft.Transform ||
					pt.Sources[0].Dtype != ft.Sources[0].Dtype || !slices.Equal(pt.Sources[0].Shape, ft.Sources[0].Shape) {
					return nil, fmt.Errorf("expert group %s: %s%s does not match the layout of %s%s", group, part.Name, suffix, first.Name, suffix)
				}
				sources[i] = pt.Sources[0]
			}
			dtype, shape := ft.OutDtype, ft.OutShape
			if dtype == "" {
				dtype = ft.Sources[0].Dtype
			}
			if shape == nil {
				shape = ft.Sources[0].Shape
			}
			if elementCount(shape) == 1 {
				// A scalar per expert ships as [] or [1]; the bank is one value per expert.
				shape = nil
			}
			bank.Tensors = append(bank.Tensors, TensorSpec{
				Name:      projection.Name + suffix,
				Sources:   sources,
				Transform: TransformStackExperts,
				Each:      ft.Transform,
				OutDtype:  dtype,
				OutShape:  append([]int32{int32(count)}, shape...),
			})
		}
		if bank.Metadata == nil {
			bank.Metadata = first.Metadata
		} else if !maps.Equal(bank.Metadata, first.Metadata) {
			return nil, fmt.Errorf("expert group %s: projections have different quantization metadata", group)
		}
	}
	return specs, nil
}

func elementCount(shape []int32) int64 {
	n := int64(1)
	for _, d := range shape {
		n *= int64(d)
	}
	return n
}

// planExpertGroup stacks each projection's per-expert weights into an
// [experts, out, in] tensor. Uniform projections share one blob; mixed
// precisions use one blob per projection so safetensors quantization metadata
// always describes the entire blob.
func planExpertGroup(groupPrefix string, tensors []SourceTensor, quantize string, policy quantizePolicy) ([]BlobSpec, error) {
	byProj := make(map[string]map[int]SourceTensor)
	for _, t := range tensors {
		idx, proj, err := parseExpertTensor(groupPrefix, t.Name)
		if err != nil {
			return nil, err
		}
		if byProj[proj] == nil {
			byProj[proj] = make(map[int]SourceTensor)
		}
		if _, ok := byProj[proj][idx]; ok {
			return nil, fmt.Errorf("expert group %s projection %s has duplicate expert index %d", groupPrefix, proj, idx)
		}
		byProj[proj][idx] = t
	}

	expertCount, err := validateExpertProjections(groupPrefix, byProj)
	if err != nil {
		return nil, err
	}
	var tensorSpecs []TensorSpec
	for _, proj := range sortedKeys(byProj) {
		experts := byProj[proj]
		base := experts[0]
		sources := make([]SourceTensor, expertCount)
		for i := range expertCount {
			e := experts[i]
			if e.Dtype != base.Dtype || !slices.Equal(e.Shape, base.Shape) {
				return nil, fmt.Errorf("expert group %s projection %s has mismatched expert layout (%s %v vs %s %v)",
					groupPrefix, proj, base.Dtype, base.Shape, e.Dtype, e.Shape)
			}
			sources[i] = e
		}

		stackedName := groupPrefix + "." + proj + ".weight"
		stackedShape := append([]int32{int32(expertCount)}, base.Shape...)
		q := ""
		if quantize != "" {
			q = policy.quantizationType(stackedName, stackedShape, quantize)
		}
		tensorSpecs = append(tensorSpecs, TensorSpec{
			Name:      stackedName,
			Sources:   sources,
			Transform: TransformStackExperts,
			Quantize:  q,
			OutDtype:  base.Dtype,
			OutShape:  stackedShape,
		})
	}
	return homogeneousExpertBlobs(groupPrefix, tensorSpecs), nil
}

func homogeneousExpertBlobs(groupPrefix string, tensors []TensorSpec) []BlobSpec {
	if len(tensors) == 0 {
		return nil
	}

	quantize := tensors[0].Quantize
	for _, tensor := range tensors[1:] {
		if tensor.Quantize != quantize {
			blobs := make([]BlobSpec, len(tensors))
			for i, tensor := range tensors {
				blobs[i] = BlobSpec{Name: tensor.Name, Tensors: []TensorSpec{tensor}}
			}
			return blobs
		}
	}

	return []BlobSpec{{Name: groupPrefix, Tensors: tensors}}
}

// parseExpertTensor splits a per-expert weight name of the form
// "<groupPrefix>.<index>.<projection>.weight" into its expert index and
// projection name.
func parseExpertTensor(groupPrefix, name string) (idx int, proj string, err error) {
	rest, ok := strings.CutPrefix(name, groupPrefix+".")
	if !ok {
		return 0, "", fmt.Errorf("expert tensor %q is not under group %q", name, groupPrefix)
	}
	rest, ok = strings.CutSuffix(rest, ".weight")
	if !ok {
		return 0, "", fmt.Errorf("expert tensor %q does not end in .weight", name)
	}
	idxStr, proj, ok := strings.Cut(rest, ".")
	if !ok {
		return 0, "", fmt.Errorf("expert tensor %q is not <index>.<projection>.weight", name)
	}
	idx, err = strconv.Atoi(idxStr)
	if err != nil {
		return 0, "", fmt.Errorf("expert tensor %q has a non-numeric expert index %q", name, idxStr)
	}
	if idx < 0 || strconv.Itoa(idx) != idxStr {
		return 0, "", fmt.Errorf("expert tensor %q has invalid expert index %q (must be a canonical non-negative integer)", name, idxStr)
	}
	return idx, proj, nil
}

// validateExpertProjections ensures source indices map directly to runtime
// slots. Every projection in a group must cover the same contiguous 0..N-1
// expert set; otherwise stacking would silently renumber or misalign experts.
func validateExpertProjections[T any](groupPrefix string, byProjection map[string]map[int]T) (int, error) {
	expectedCount := -1
	for _, projection := range sortedKeys(byProjection) {
		experts := byProjection[projection]
		for index := range len(experts) {
			if _, ok := experts[index]; !ok {
				return 0, fmt.Errorf("expert group %s projection %s is missing expert index %d", groupPrefix, projection, index)
			}
		}
		if expectedCount >= 0 && len(experts) != expectedCount {
			return 0, fmt.Errorf("expert group %s projection %s has %d experts, want %d to match the other projections", groupPrefix, projection, len(experts), expectedCount)
		}
		expectedCount = len(experts)
	}
	if expectedCount < 0 {
		return 0, nil
	}
	return expectedCount, nil
}

// perExpertGroup reports whether name is a per-expert weight that must be
// stacked and returns its group prefix: a "...experts" element followed by an
// expert index and a projection, inside a "layers.<n>" block, as in
// "<prefix>.layers.3.mlp.experts.7.gate_proj.weight" or
// "<prefix>.layers.3.experts.7.gate_proj.weight". An already-stacked expert
// tensor (one tensor covering all experts, e.g.
// "<layer>.mlp.experts.gate_up_proj.weight" as qwen3.5 and gemma4 ship it) is
// not per-expert: it is quantized as an ordinary 3D tensor and the runtime
// splits/uses it directly.
func perExpertGroup(name string) (string, bool) {
	parts := strings.Split(name, ".")
	if len(parts) < 5 || parts[len(parts)-1] != "weight" {
		return "", false
	}
	inLayer := false
	for i := 0; i+3 < len(parts); i++ {
		if parts[i] == "layers" && isInteger(parts[i+1]) {
			inLayer = true
			continue
		}
		// Any integer groups here, so a malformed index ("01", "-1") reaches
		// parseExpertTensor and fails instead of becoming a standalone blob.
		if inLayer && strings.HasSuffix(parts[i], "experts") && isInteger(parts[i+1]) {
			return strings.Join(parts[:i+1], "."), true
		}
	}
	return "", false
}

func isInteger(s string) bool {
	_, err := strconv.Atoi(s)
	return err == nil
}

func sortedTensorNames(inv Inventory) []string {
	return sortedKeys(inv.Tensors)
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
