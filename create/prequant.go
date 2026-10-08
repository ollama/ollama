package create

import "strings"

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
// companions "<base>.weight.scale", ".bias", ".global_scale", and any
// producer-provided activation scales.
type prequantPattern struct {
	name string

	weightSuffix string // source suffix identifying the weight (".weight" or ".weight_packed")
	repackWeight bool   // repack a U8 fp4 weight into U32 words

	scaleSuffix    string // required per-block / affine scale companion
	scaleRelabelU8 bool   // relabel an F8_E4M3 scale as U8 for the loader

	biasSuffix string // optional bias / zero-point companion ("" if none)

	globalSuffix     string // optional global-scale companion ("" if none)
	globalReciprocal bool   // producer stores weight and activation global scales inverted

	activationScaleSuffixes []string // optional activation scales preserved beside the weight

	forceQuantType   string // override the blob's quant_type metadata
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
		name:                    "compressed-tensors-nvfp4",
		weightSuffix:            ".weight_packed",
		repackWeight:            true,
		scaleSuffix:             ".weight_scale",
		scaleRelabelU8:          true,
		globalSuffix:            ".weight_global_scale",
		globalReciprocal:        true,
		activationScaleSuffixes: []string{".input_scale", ".input_global_scale"},
		forceQuantType:          "nvfp4",
		defaultGroupSize:        "16",
	},
	{
		name:                    "modelopt-nvfp4",
		weightSuffix:            ".weight",
		repackWeight:            true,
		scaleSuffix:             ".weight_scale",
		scaleRelabelU8:          true,
		globalSuffix:            ".weight_scale_2",
		activationScaleSuffixes: []string{".input_scale", ".input_global_scale"},
		forceQuantType:          "nvfp4",
	},
}

// planPrequantized plans an already-quantized source: each weight is fused with
// its scale companions into one blob, companions are not emitted on their own,
// and any remaining tensors (norms, embeddings) pass through at source
// precision.
func planPrequantized(inv Inventory) ([]BlobSpec, error) {
	fused := make(map[string]BlobSpec)
	consumed := make(map[string]bool)
	for _, name := range sortedTensorNames(inv) {
		spec, sources, ok := matchPrequant(name, inv)
		if !ok {
			continue
		}
		fused[name] = spec
		for _, s := range sources {
			consumed[s] = true
		}
	}

	specs := make([]BlobSpec, 0, len(inv.Tensors))
	for _, name := range sortedTensorNames(inv) {
		if spec, ok := fused[name]; ok {
			specs = append(specs, spec)
			continue
		}
		if consumed[name] {
			continue
		}
		t := inv.Tensors[name]
		specs = append(specs, BlobSpec{Name: name, Tensors: []TensorSpec{{Name: name, Sources: []SourceTensor{t}}}})
	}
	return specs, nil
}

// matchPrequant returns the fused blob for a weight tensor if it matches a
// prequantized producer, along with the source names it consumes. It returns
// ok=false when name is not a prequantized weight (a companion or a plain
// tensor).
func matchPrequant(name string, inv Inventory) (BlobSpec, []string, bool) {
	for _, p := range prequantPatterns {
		base, ok := strings.CutSuffix(name, p.weightSuffix)
		if !ok {
			continue
		}
		scaleSrc := base + p.scaleSuffix
		if !inv.Has(scaleSrc) {
			continue
		}

		outWeight := base + ".weight"
		weight := inv.Tensors[name]
		var tensors []TensorSpec
		var consumed []string

		weightTensor := TensorSpec{Name: outWeight, Sources: []SourceTensor{weight}}
		if p.repackWeight && strings.EqualFold(weight.Dtype, "U8") && len(weight.Shape) == 2 {
			weightTensor.Transform = TransformRepackFP4
			weightTensor.OutDtype = "U32"
			weightTensor.OutShape = []int32{weight.Shape[0], weight.Shape[1] / 4}
		}
		tensors = append(tensors, weightTensor)

		scale := inv.Tensors[scaleSrc]
		scaleTensor := TensorSpec{Name: outWeight + ".scale", Sources: []SourceTensor{scale}}
		if p.scaleRelabelU8 && isE4M3Dtype(scale.Dtype) {
			scaleTensor.Transform = TransformRelabelU8
			scaleTensor.OutDtype = "U8"
		}
		tensors = append(tensors, scaleTensor)
		consumed = append(consumed, scaleSrc)

		if p.biasSuffix != "" {
			if biasSrc := base + p.biasSuffix; inv.Has(biasSrc) {
				tensors = append(tensors, TensorSpec{Name: outWeight + ".bias", Sources: []SourceTensor{inv.Tensors[biasSrc]}})
				consumed = append(consumed, biasSrc)
			}
		}

		if p.globalSuffix != "" {
			if gSrc := base + p.globalSuffix; inv.Has(gSrc) {
				globalSource := inv.Tensors[gSrc]
				global := TensorSpec{Name: outWeight + ".global_scale", Sources: []SourceTensor{globalSource}, Transform: f32ScaleTransform(globalSource)}
				if p.globalReciprocal {
					global.Transform = TransformReciprocalF32
				}
				tensors = append(tensors, global)
				consumed = append(consumed, gSrc)
			}
		}

		for _, suffix := range p.activationScaleSuffixes {
			if source := base + suffix; inv.Has(source) {
				scale := inv.Tensors[source]
				transform := f32ScaleTransform(scale)
				// Like weight_global_scale, compressed-tensors' input_global_scale
				// is a divisor. Store both as dequantization multipliers, matching
				// ModelOpt. input_scale is a local scale and is not inverted.
				if p.globalReciprocal && suffix == ".input_global_scale" {
					transform = TransformReciprocalF32
				}
				tensors = append(tensors, TensorSpec{
					Name:      outWeight + suffix,
					Sources:   []SourceTensor{scale},
					Transform: transform,
				})
				consumed = append(consumed, source)
			}
		}

		return BlobSpec{Name: outWeight, Tensors: tensors, Metadata: prequantMetadata(inv, p)}, consumed, true
	}
	return BlobSpec{}, nil, false
}

func f32ScaleTransform(t SourceTensor) Transform {
	for _, dim := range t.Shape {
		if dim != 1 {
			return TransformF32
		}
	}
	return TransformScalarF32
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
