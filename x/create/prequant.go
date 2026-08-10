package create

import (
	"fmt"
	"slices"
	"sort"
	"strconv"
	"strings"
)

// prequantPattern describes how one producer packs an already-quantized weight
// and its scale companions into safetensors files, and how to fuse them into
// the single blob our loader reads. Producers differ only in tensor names and a
// few per-field transforms; expressing them as table rows keeps those
// differences visible and prevents the per-producer drift the old separate code
// paths suffered (for example the global scale being stored as-is by one
// producer and inverted by another).
//
// All suffixes are relative to the base — the source weight name minus its
// weight suffix. The fused blob is always named "<base>.weight", with
// companions "<base>.weight.scale", ".bias", and ".global_scale".
type prequantPattern struct {
	name string

	weightSuffix     string   // source suffix identifying the weight (".weight" or ".weight_packed")
	weightDtypes     []string // optional accepted source dtypes
	repackWeight     bool     // repack a U8 fp4 weight into U32 words
	relabelWeightU32 bool     // relabel an I32 packed container as U32

	scaleSuffix    string   // required per-block / affine scale companion
	scaleDtypes    []string // optional accepted scale dtypes
	scaleRelabelU8 bool     // relabel an F8_E4M3 scale as U8 for the loader

	biasSuffix       string // optional bias / zero-point companion ("" if none)
	deriveInt4QBias  bool   // derive MLX affine bias=-8*scale for symmetric INT4
	requiredSuffixes []string

	globalSuffix     string // optional global-scale companion ("" if none)
	globalReciprocal bool   // store the global scale as its reciprocal

	ignoreSuffixes []string // companions consumed but not written (e.g. activation scales)

	forceQuantType   string // override the blob's quant_type metadata
	requireQuantType string // pattern applies only when config metadata has this type
	defaultGroupSize string // set group_size metadata only when the config did not
}

// prequantPatterns is consulted in order; the first whose weight suffix matches
// and whose required scale companion is present wins. MLX and ModelOpt both use
// a ".weight" weight, but their scale companions (".scales" vs ".weight_scale")
// are mutually exclusive, so the order between them does not matter.
var prequantPatterns = []prequantPattern{
	{
		name:         "mlx",
		weightSuffix: ".weight",
		scaleSuffix:  ".scales",
		biasSuffix:   ".biases",
	},
	{
		name:             "compressed-tensors-int4",
		weightSuffix:     ".weight_packed",
		weightDtypes:     []string{"I32"},
		relabelWeightU32: true,
		scaleSuffix:      ".weight_scale",
		scaleDtypes:      []string{"BF16", "F16", "F32"},
		deriveInt4QBias:  true,
		requiredSuffixes: []string{".weight_shape"},
		requireQuantType: "int4",
	},
	{
		name:             "compressed-tensors-nvfp4",
		weightSuffix:     ".weight_packed",
		weightDtypes:     []string{"U8"},
		repackWeight:     true,
		scaleSuffix:      ".weight_scale",
		scaleDtypes:      []string{"F8_E4M3", "F8_E4M3FN"},
		scaleRelabelU8:   true,
		globalSuffix:     ".weight_global_scale",
		globalReciprocal: true,
		ignoreSuffixes:   []string{".input_scale", ".input_global_scale"},
		forceQuantType:   "nvfp4",
		defaultGroupSize: "16",
	},
	{
		name:           "modelopt-nvfp4",
		weightSuffix:   ".weight",
		weightDtypes:   []string{"U8"},
		repackWeight:   true,
		scaleSuffix:    ".weight_scale",
		scaleDtypes:    []string{"F8_E4M3", "F8_E4M3FN"},
		scaleRelabelU8: true,
		globalSuffix:   ".weight_scale_2",
		ignoreSuffixes: []string{".input_scale", ".input_global_scale"},
		forceQuantType: "nvfp4",
	},
}

// planPrequantized plans an already-quantized source: each weight is fused with
// its scale companions into one blob, companions are not emitted on their own,
// and any remaining tensors (norms, embeddings) pass through at source
// precision.
func planPrequantized(inv Inventory) ([]BlobSpec, error) {
	expertSpecs, consumed, err := planPrequantizedExpertGroups(inv)
	if err != nil {
		return nil, err
	}

	fused := make(map[string]BlobSpec)
	for _, name := range sortedTensorNames(inv) {
		if consumed[name] {
			continue
		}
		spec, sources, ok, err := matchPrequant(name, inv)
		if err != nil {
			return nil, err
		}
		if !ok {
			continue
		}
		fused[name] = spec
		for _, s := range sources {
			consumed[s] = true
		}
	}

	specs := make([]BlobSpec, 0, len(expertSpecs)+len(inv.Tensors))
	specs = append(specs, expertSpecs...)
	for _, name := range sortedTensorNames(inv) {
		if spec, ok := fused[name]; ok {
			specs = append(specs, spec)
			continue
		}
		if consumed[name] {
			continue
		}
		t := inv.Tensors[name]
		if strings.HasSuffix(name, ".weight_packed") && strings.EqualFold(t.Dtype, "I32") {
			return nil, fmt.Errorf("prequantized tensor %s uses unsupported packed integer metadata; compressed-tensors INT4 requires symmetric group quantization with weight_shape", name)
		}
		specs = append(specs, BlobSpec{Name: name, Tensors: []TensorSpec{{Name: name, Sources: []SourceTensor{t}}}})
	}
	return specs, nil
}

// planPrequantizedExpertGroups turns per-expert compressed-tensors INT4
// triples into the stacked layout consumed by GatherQMM. The packed words and
// scales are concatenated without requantization; only the affine qbias tensor
// is derived from the scales.
func planPrequantizedExpertGroups(inv Inventory) ([]BlobSpec, map[string]bool, error) {
	type packedExpert struct {
		idx      int
		proj     string
		weight   SourceTensor
		scale    SourceTensor
		metadata map[string]string
		sources  []string
	}

	groups := make(map[string][]packedExpert)
	for _, name := range sortedTensorNames(inv) {
		if !strings.HasSuffix(name, ".weight_packed") || !strings.EqualFold(inv.Tensors[name].Dtype, "I32") {
			continue
		}
		base := strings.TrimSuffix(name, ".weight_packed")
		normalized := base + ".weight"
		groupPrefix, ok := perExpertGroup(normalized)
		if !ok {
			continue
		}

		spec, companions, ok, err := matchPrequant(name, inv)
		if err != nil {
			return nil, nil, err
		}
		if !ok || !strings.EqualFold(spec.Metadata["quant_type"], "int4") {
			continue
		}
		idx, proj, err := parseExpertTensor(groupPrefix, normalized)
		if err != nil {
			return nil, nil, err
		}
		groups[groupPrefix] = append(groups[groupPrefix], packedExpert{
			idx:      idx,
			proj:     proj,
			weight:   inv.Tensors[name],
			scale:    inv.Tensors[base+".weight_scale"],
			metadata: spec.Metadata,
			sources:  append([]string{name}, companions...),
		})
	}

	consumed := make(map[string]bool)
	var specs []BlobSpec
	for _, groupPrefix := range sortedKeys(groups) {
		byProj := make(map[string][]packedExpert)
		for _, expert := range groups[groupPrefix] {
			byProj[expert.proj] = append(byProj[expert.proj], expert)
		}

		var tensors []TensorSpec
		var metadata map[string]string
		for _, proj := range sortedKeys(byProj) {
			experts := byProj[proj]
			sort.Slice(experts, func(i, j int) bool { return experts[i].idx < experts[j].idx })
			base := experts[0]
			weightSources := make([]SourceTensor, len(experts))
			scaleSources := make([]SourceTensor, len(experts))
			for i, expert := range experts {
				if expert.idx != i {
					return nil, nil, fmt.Errorf("expert group %s projection %s has indices 0..%d with missing or duplicate index at %d", groupPrefix, proj, len(experts)-1, expert.idx)
				}
				if !strings.EqualFold(expert.weight.Dtype, base.weight.Dtype) || !slices.Equal(expert.weight.Shape, base.weight.Shape) {
					return nil, nil, fmt.Errorf("expert group %s projection %s has mismatched packed weight layout (%s %v vs %s %v)", groupPrefix, proj, base.weight.Dtype, base.weight.Shape, expert.weight.Dtype, expert.weight.Shape)
				}
				if !strings.EqualFold(expert.scale.Dtype, base.scale.Dtype) || !slices.Equal(expert.scale.Shape, base.scale.Shape) {
					return nil, nil, fmt.Errorf("expert group %s projection %s has mismatched scale layout (%s %v vs %s %v)", groupPrefix, proj, base.scale.Dtype, base.scale.Shape, expert.scale.Dtype, expert.scale.Shape)
				}
				if expert.metadata["quant_type"] != base.metadata["quant_type"] || expert.metadata["group_size"] != base.metadata["group_size"] {
					return nil, nil, fmt.Errorf("expert group %s projection %s has mismatched quantization metadata", groupPrefix, proj)
				}
				weightSources[i] = expert.weight
				scaleSources[i] = expert.scale
				for _, source := range expert.sources {
					consumed[source] = true
				}
			}

			outWeight := groupPrefix + "." + proj + ".weight"
			weightShape := append([]int32{int32(len(experts))}, base.weight.Shape...)
			scaleShape := append([]int32{int32(len(experts))}, base.scale.Shape...)
			tensors = append(tensors,
				TensorSpec{
					Name:      outWeight,
					Sources:   weightSources,
					Transform: TransformStackExperts,
					OutDtype:  "U32",
					OutShape:  weightShape,
				},
				TensorSpec{
					Name:      outWeight + ".scale",
					Sources:   scaleSources,
					Transform: TransformStackExperts,
					OutDtype:  base.scale.Dtype,
					OutShape:  scaleShape,
				},
				TensorSpec{
					Name:      outWeight + ".bias",
					Sources:   scaleSources,
					Transform: TransformInt4SymmetricQBias,
					OutDtype:  base.scale.Dtype,
					OutShape:  scaleShape,
				},
			)
			if metadata == nil {
				metadata = base.metadata
			}
		}
		specs = append(specs, BlobSpec{Name: groupPrefix, Tensors: tensors, Metadata: metadata})
	}
	return specs, consumed, nil
}

// matchPrequant returns the fused blob for a weight tensor if it matches a
// prequantized producer, along with the source names it consumes. It returns
// ok=false when name is not a prequantized weight (a companion or a plain
// tensor).
func matchPrequant(name string, inv Inventory) (BlobSpec, []string, bool, error) {
patterns:
	for _, p := range prequantPatterns {
		base, ok := strings.CutSuffix(name, p.weightSuffix)
		if !ok {
			continue
		}
		weight := inv.Tensors[name]
		if !dtypeAccepted(weight.Dtype, p.weightDtypes) {
			continue
		}
		scaleSrc := base + p.scaleSuffix
		if !inv.Has(scaleSrc) {
			continue
		}
		scale := inv.Tensors[scaleSrc]
		if !dtypeAccepted(scale.Dtype, p.scaleDtypes) {
			continue
		}

		metadata := prequantMetadata(inv, p)
		if p.requireQuantType != "" && !strings.EqualFold(metadata["quant_type"], p.requireQuantType) {
			continue
		}
		for _, suffix := range p.requiredSuffixes {
			if !inv.Has(base + suffix) {
				if p.name == "compressed-tensors-int4" {
					return BlobSpec{}, nil, false, fmt.Errorf("prequantized tensor %s is missing required companion %s", name, base+suffix)
				}
				continue patterns
			}
		}
		if p.name == "compressed-tensors-int4" {
			groupSize, err := strconv.Atoi(metadata["group_size"])
			if err != nil || groupSize <= 0 {
				return BlobSpec{}, nil, false, fmt.Errorf("prequantized tensor %s has invalid compressed-tensors INT4 group_size %q", name, metadata["group_size"])
			}
			if err := validateCompressedInt4Layout(weight, scale, inv.Tensors[base+".weight_shape"], groupSize); err != nil {
				return BlobSpec{}, nil, false, fmt.Errorf("prequantized tensor %s: %w", name, err)
			}
		}

		outWeight := base + ".weight"
		var tensors []TensorSpec
		var consumed []string

		weightTensor := TensorSpec{Name: outWeight, Sources: []SourceTensor{weight}}
		if p.repackWeight && strings.EqualFold(weight.Dtype, "U8") && len(weight.Shape) == 2 {
			weightTensor.Transform = TransformRepackFP4
			weightTensor.OutDtype = "U32"
			weightTensor.OutShape = []int32{weight.Shape[0], weight.Shape[1] / 4}
		} else if p.relabelWeightU32 {
			weightTensor.Transform = TransformRelabelU32
			weightTensor.OutDtype = "U32"
		}
		tensors = append(tensors, weightTensor)

		scaleTensor := TensorSpec{Name: outWeight + ".scale", Sources: []SourceTensor{scale}}
		if p.scaleRelabelU8 && isE4M3Dtype(scale.Dtype) {
			scaleTensor.Transform = TransformRelabelU8
			scaleTensor.OutDtype = "U8"
		}
		tensors = append(tensors, scaleTensor)
		consumed = append(consumed, scaleSrc)
		if p.deriveInt4QBias {
			tensors = append(tensors, TensorSpec{
				Name:      outWeight + ".bias",
				Sources:   []SourceTensor{scale},
				Transform: TransformInt4SymmetricQBias,
				OutDtype:  scale.Dtype,
			})
		}

		if p.biasSuffix != "" {
			if biasSrc := base + p.biasSuffix; inv.Has(biasSrc) {
				tensors = append(tensors, TensorSpec{Name: outWeight + ".bias", Sources: []SourceTensor{inv.Tensors[biasSrc]}})
				consumed = append(consumed, biasSrc)
			}
		}

		if p.globalSuffix != "" {
			if gSrc := base + p.globalSuffix; inv.Has(gSrc) {
				global := TensorSpec{Name: outWeight + ".global_scale", Sources: []SourceTensor{inv.Tensors[gSrc]}, Transform: TransformScalarF32}
				if p.globalReciprocal {
					global.Transform = TransformReciprocalF32
				}
				tensors = append(tensors, global)
				consumed = append(consumed, gSrc)
			}
		}

		for _, suf := range p.ignoreSuffixes {
			if s := base + suf; inv.Has(s) {
				consumed = append(consumed, s)
			}
		}
		for _, suf := range p.requiredSuffixes {
			consumed = append(consumed, base+suf)
		}

		return BlobSpec{Name: outWeight, Tensors: tensors, Metadata: metadata}, consumed, true, nil
	}
	return BlobSpec{}, nil, false, nil
}

func dtypeAccepted(dtype string, accepted []string) bool {
	if len(accepted) == 0 {
		return true
	}
	for _, candidate := range accepted {
		if strings.EqualFold(dtype, candidate) {
			return true
		}
	}
	return false
}

func validateCompressedInt4Layout(weight, scale, shape SourceTensor, groupSize int) error {
	if len(weight.Shape) < 2 || len(weight.Shape) != len(scale.Shape) {
		return fmt.Errorf("packed weight shape %v and scale shape %v must have the same rank >= 2", weight.Shape, scale.Shape)
	}
	for i := 0; i < len(weight.Shape)-1; i++ {
		if weight.Shape[i] != scale.Shape[i] {
			return fmt.Errorf("packed weight shape %v and scale shape %v disagree at dimension %d", weight.Shape, scale.Shape, i)
		}
	}
	packedValues := int64(weight.Shape[len(weight.Shape)-1]) * 8
	scaledValues := int64(scale.Shape[len(scale.Shape)-1]) * int64(groupSize)
	if packedValues != scaledValues {
		return fmt.Errorf("packed weight shape %v describes %d values per row but scale shape %v with group_size=%d describes %d", weight.Shape, packedValues, scale.Shape, groupSize, scaledValues)
	}
	if !strings.EqualFold(shape.Dtype, "I64") || !slices.Equal(shape.Shape, []int32{int32(len(weight.Shape))}) {
		return fmt.Errorf("weight_shape must be I64 [%d], got %s %v", len(weight.Shape), shape.Dtype, shape.Shape)
	}
	return nil
}

// prequantMetadata builds the fused blob's metadata: the source config's quant
// metadata, with the pattern's quant_type override and group_size default
// applied. Returns nil when there is nothing to record.
func prequantMetadata(inv Inventory, p prequantPattern) map[string]string {
	md := make(map[string]string)
	for k, v := range inv.Config.QuantMetadata() {
		md[k] = v
	}
	if p.forceQuantType != "" {
		md["quant_type"] = p.forceQuantType
	}
	if p.defaultGroupSize != "" {
		if _, ok := md["group_size"]; !ok {
			md["group_size"] = p.defaultGroupSize
		}
	}
	if len(md) == 0 {
		return nil
	}
	return md
}
