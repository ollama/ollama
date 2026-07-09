package create

import (
	"fmt"
	"slices"
	"sort"
	"strings"
)

// planBlockFP8 plans an HF block-FP8 source. MLX has no FP8 tensor type, so
// every FP8 weight is decoded to BF16 using its block scale and then quantized
// to the target (mxfp8); a weight the policy declines is still decoded and kept
// at BF16 (it is never stored as FP8). Everything else passes through at source
// precision.
func planBlockFP8(inv Inventory, target string, policy quantizePolicy) ([]BlobSpec, error) {
	// The scale companion of each FP8 weight is folded into that weight's
	// blob, so it is not emitted on its own.
	consumed := make(map[string]bool)
	for _, name := range sortedTensorNames(inv) {
		if isFP8Weight(inv, name) {
			if scale, ok := fp8ScaleFor(inv, name); ok {
				consumed[scale] = true
			}
		}
	}

	groups := make(map[string][]SourceTensor)
	fp8Groups := make(map[string][]SourceTensor)
	specs := make([]BlobSpec, 0, len(inv.Tensors))
	for _, name := range sortedTensorNames(inv) {
		if consumed[name] {
			continue
		}
		t := inv.Tensors[name]

		if isFP8Weight(inv, name) {
			// Disjoint per-expert FP8 weights are stacked, decoded, and
			// quantized together by planFP8ExpertGroup; an already-stacked (3D)
			// FP8 expert tensor falls through to the single-tensor decode below.
			if gp, perExpert := perExpertGroup(name); perExpert {
				fp8Groups[gp] = append(fp8Groups[gp], t)
				continue
			}
			scaleName, ok := fp8ScaleFor(inv, name)
			if !ok {
				return nil, fmt.Errorf("fp8 weight %q has no scale companion", name)
			}
			specs = append(specs, BlobSpec{
				Name: name,
				Tensors: []TensorSpec{{
					Name:      name,
					Sources:   []SourceTensor{t, inv.Tensors[scaleName]},
					Transform: TransformDecodeFP8,
					Quantize:  policy.quantizationType(name, t.Shape, target),
					OutDtype:  "BF16",
					OutShape:  t.Shape,
				}},
			})
			continue
		}

		if gp, ok := perExpertGroup(name); ok {
			groups[gp] = append(groups[gp], t)
			continue
		}

		specs = append(specs, BlobSpec{
			Name:    name,
			Tensors: []TensorSpec{{Name: name, Sources: []SourceTensor{t}}},
		})
	}

	for _, gp := range sortedKeys(groups) {
		spec, err := planExpertGroup(gp, groups[gp], "", policy)
		if err != nil {
			return nil, err
		}
		specs = append(specs, spec)
	}
	for _, gp := range sortedKeys(fp8Groups) {
		spec, err := planFP8ExpertGroup(gp, fp8Groups[gp], inv, target, policy)
		if err != nil {
			return nil, err
		}
		specs = append(specs, spec)
	}
	return specs, nil
}

// planFP8ExpertGroup packs a layer's disjoint per-expert block-FP8 weights into
// one blob: the experts of each projection are stacked into [experts, out, in],
// dequantized from FP8 with their block scales, and quantized per the policy.
// The stacking, decode, and quantize all run on the MLX writer thread; the
// planner only groups and orders the source weights and their scale companions.
func planFP8ExpertGroup(groupPrefix string, tensors []SourceTensor, inv Inventory, target string, policy quantizePolicy) (BlobSpec, error) {
	type expert struct {
		idx    int
		weight SourceTensor
		scale  SourceTensor
	}
	byProj := make(map[string][]expert)
	for _, t := range tensors {
		idx, proj, err := parseExpertTensor(groupPrefix, t.Name)
		if err != nil {
			return BlobSpec{}, err
		}
		scaleName, ok := fp8ScaleFor(inv, t.Name)
		if !ok {
			return BlobSpec{}, fmt.Errorf("fp8 expert weight %q has no scale companion", t.Name)
		}
		byProj[proj] = append(byProj[proj], expert{idx: idx, weight: t, scale: inv.Tensors[scaleName]})
	}

	spec := BlobSpec{Name: groupPrefix}
	for _, proj := range sortedKeys(byProj) {
		experts := byProj[proj]
		sort.Slice(experts, func(i, j int) bool { return experts[i].idx < experts[j].idx })

		base := experts[0].weight
		baseScale := experts[0].scale
		// Sources are the N weights followed by the N scales, in expert order,
		// matching what TransformDecodeStackFP8 expects.
		sources := make([]SourceTensor, 0, 2*len(experts))
		scales := make([]SourceTensor, 0, len(experts))
		for _, e := range experts {
			if e.weight.Dtype != base.Dtype || !slices.Equal(e.weight.Shape, base.Shape) {
				return BlobSpec{}, fmt.Errorf("fp8 expert group %s projection %s has mismatched weight layout (%s %v vs %s %v)",
					groupPrefix, proj, base.Dtype, base.Shape, e.weight.Dtype, e.weight.Shape)
			}
			if e.scale.Dtype != baseScale.Dtype || !slices.Equal(e.scale.Shape, baseScale.Shape) {
				return BlobSpec{}, fmt.Errorf("fp8 expert group %s projection %s has mismatched scale layout (%s %v vs %s %v)",
					groupPrefix, proj, baseScale.Dtype, baseScale.Shape, e.scale.Dtype, e.scale.Shape)
			}
			sources = append(sources, e.weight)
			scales = append(scales, e.scale)
		}
		sources = append(sources, scales...)

		stackedName := groupPrefix + "." + proj + ".weight"
		stackedShape := append([]int32{int32(len(experts))}, base.Shape...)
		spec.Tensors = append(spec.Tensors, TensorSpec{
			Name:      stackedName,
			Sources:   sources,
			Transform: TransformDecodeStackFP8,
			Quantize:  policy.quantizationType(stackedName, stackedShape, target),
			OutDtype:  base.Dtype,
			OutShape:  stackedShape,
		})
	}
	return spec, nil
}

// isFP8Weight reports whether name is an F8_E4M3 weight with a block-scale
// companion (the form that must be decoded before use).
func isFP8Weight(inv Inventory, name string) bool {
	t, ok := inv.Tensors[name]
	if !ok || !strings.HasSuffix(name, ".weight") || !isE4M3Dtype(t.Dtype) {
		return false
	}
	_, ok = fp8ScaleFor(inv, name)
	return ok
}

// fp8ScaleFor returns the block-scale companion name for an FP8 weight,
// preferring "_scale_inv" over "_scale" (matching the source conventions).
func fp8ScaleFor(inv Inventory, weightName string) (string, bool) {
	for _, suffix := range []string{"_scale_inv", "_scale"} {
		if s := weightName + suffix; inv.Has(s) {
			return s, true
		}
	}
	return "", false
}
