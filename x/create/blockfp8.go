package create

import (
	"fmt"
	"slices"
	"sort"
	"strings"
)

// planBlockFP8 plans an HF block-FP8 source. MLX has no FP8 tensor type, so
// every FP8 weight becomes mxfp8 or BF16. When the block scales are UE8M0
// exponents the mxfp8 conversion is exact and purely byte-level: E4M3 codes
// are kept bit-for-bit and each 128x128 block exponent is replicated into the
// per-32-value group scales (both formats are E4M3 x 2^k, and every group
// lies inside one block). Weights the policy declines, and weights whose
// scales are not UE8M0, are decoded to BF16 with their block scales (and then
// re-quantized to mxfp8 when the policy asks for it). Everything else passes
// through at source precision.
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
			quantize := policy.quantizationType(name, t.Shape, target)
			if quantize == "mxfp8" && canRepackBlockFP8(t, inv.Tensors[scaleName]) {
				specs = append(specs, BlobSpec{
					Name:     name,
					Tensors:  repackBlockFP8Tensors(name, []SourceTensor{t}, []SourceTensor{inv.Tensors[scaleName]}, t.Shape),
					Metadata: mxfp8BlobMetadata(),
				})
				continue
			}
			specs = append(specs, BlobSpec{
				Name: name,
				Tensors: []TensorSpec{{
					Name:      name,
					Sources:   []SourceTensor{t, inv.Tensors[scaleName]},
					Transform: TransformDecodeFP8,
					Quantize:  quantize,
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
		groupSpecs, err := planExpertGroup(gp, groups[gp], "", policy)
		if err != nil {
			return nil, err
		}
		specs = append(specs, groupSpecs...)
	}
	for _, gp := range sortedKeys(fp8Groups) {
		groupSpecs, err := planFP8ExpertGroup(gp, fp8Groups[gp], inv, target, policy)
		if err != nil {
			return nil, err
		}
		specs = append(specs, groupSpecs...)
	}
	return specs, nil
}

// planFP8ExpertGroup packs a layer's disjoint per-expert block-FP8 weights into
// one blob: the experts of each projection are stacked into [experts, out, in],
// dequantized from FP8 with their block scales, and quantized per the policy.
// The stacking, decode, and quantize all run on the MLX writer thread; the
// planner only groups and orders the source weights and their scale companions.
func planFP8ExpertGroup(groupPrefix string, tensors []SourceTensor, inv Inventory, target string, policy quantizePolicy) ([]BlobSpec, error) {
	type expert struct {
		idx    int
		weight SourceTensor
		scale  SourceTensor
	}
	byProj := make(map[string][]expert)
	for _, t := range tensors {
		idx, proj, err := parseExpertTensor(groupPrefix, t.Name)
		if err != nil {
			return nil, err
		}
		scaleName, ok := fp8ScaleFor(inv, t.Name)
		if !ok {
			return nil, fmt.Errorf("fp8 expert weight %q has no scale companion", t.Name)
		}
		byProj[proj] = append(byProj[proj], expert{idx: idx, weight: t, scale: inv.Tensors[scaleName]})
	}

	var tensorSpecs []TensorSpec
	repackable := true
	type plannedProj struct {
		name    string
		shape   []int32
		weights []SourceTensor
		scales  []SourceTensor
	}
	var planned []plannedProj
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
				return nil, fmt.Errorf("fp8 expert group %s projection %s has mismatched weight layout (%s %v vs %s %v)",
					groupPrefix, proj, base.Dtype, base.Shape, e.weight.Dtype, e.weight.Shape)
			}
			if e.scale.Dtype != baseScale.Dtype || !slices.Equal(e.scale.Shape, baseScale.Shape) {
				return nil, fmt.Errorf("fp8 expert group %s projection %s has mismatched scale layout (%s %v vs %s %v)",
					groupPrefix, proj, baseScale.Dtype, baseScale.Shape, e.scale.Dtype, e.scale.Shape)
			}
			sources = append(sources, e.weight)
			scales = append(scales, e.scale)
		}
		sources = append(sources, scales...)

		stackedName := groupPrefix + "." + proj + ".weight"
		stackedShape := append([]int32{int32(len(experts))}, base.Shape...)
		quantize := policy.quantizationType(stackedName, stackedShape, target)
		if quantize != "mxfp8" || !canRepackBlockFP8(base, baseScale) {
			repackable = false
		}
		planned = append(planned, plannedProj{name: stackedName, shape: stackedShape, weights: sources[:len(experts)], scales: scales})
		tensorSpecs = append(tensorSpecs, TensorSpec{
			Name:      stackedName,
			Sources:   sources,
			Transform: TransformDecodeStackFP8,
			Quantize:  quantize,
			OutDtype:  base.Dtype,
			OutShape:  stackedShape,
		})
	}

	// The lossless byte repack applies only when every projection of the group
	// qualifies, so the blob's quant metadata describes all of its tensors.
	if repackable {
		var repacked []TensorSpec
		for _, proj := range planned {
			repacked = append(repacked, repackBlockFP8Tensors(proj.name, proj.weights, proj.scales, proj.shape)...)
		}
		return []BlobSpec{{Name: groupPrefix, Tensors: repacked, Metadata: mxfp8BlobMetadata()}}, nil
	}
	return homogeneousExpertBlobs(groupPrefix, tensorSpecs), nil
}

// canRepackBlockFP8 reports whether an FP8 weight and its block scale can be
// converted to mxfp8 losslessly at the byte level: UE8M0 exponent scales over
// full 128x128 blocks of an E4M3 weight whose rows split evenly into 32-value
// groups.
func canRepackBlockFP8(weight, scale SourceTensor) bool {
	if !isE4M3Dtype(weight.Dtype) || !isE8M0Dtype(scale.Dtype) {
		return false
	}
	rank := len(weight.Shape)
	if rank < 2 || len(scale.Shape) != rank {
		return false
	}
	rows, cols := weight.Shape[rank-2], weight.Shape[rank-1]
	if cols%32 != 0 {
		return false
	}
	sr := (rows + 127) / 128
	sc := (cols + 127) / 128
	want := append(append([]int32(nil), weight.Shape[:rank-2]...), sr, sc)
	return slices.Equal(scale.Shape, want)
}

// repackBlockFP8Tensors builds the two byte-level TensorSpecs that convert a
// block-FP8 weight (or a stack of per-expert weights) to mxfp8 exactly: the
// E4M3 bytes relabeled as packed U32 words, and the block scales expanded to
// per-group UE8M0 bytes.
func repackBlockFP8Tensors(name string, weights, scales []SourceTensor, outShape []int32) []TensorSpec {
	rank := len(outShape)
	packedShape := append([]int32(nil), outShape...)
	packedShape[rank-1] /= 4
	scaleShape := append([]int32(nil), outShape...)
	scaleShape[rank-1] /= 32

	weightTransform := TransformRelabelU32
	if len(weights) > 1 {
		weightTransform = TransformStackExperts
	}
	return []TensorSpec{
		{
			Name:      name,
			Sources:   weights,
			Transform: weightTransform,
			OutDtype:  "U32",
			OutShape:  packedShape,
		},
		{
			Name:      name + ".scale",
			Sources:   scales,
			Transform: TransformBlockFP8GroupScales,
			OutDtype:  "U8",
			OutShape:  scaleShape,
		},
	}
}

// mxfp8BlobMetadata is the safetensors metadata recorded on a losslessly
// repacked block-FP8 blob, matching what the MLX mxfp8 quantizer writes.
func mxfp8BlobMetadata() map[string]string {
	return map[string]string{"quant_type": "mxfp8", "group_size": "32"}
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
