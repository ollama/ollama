package create

import (
	"fmt"
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

	weightSuffix string // source suffix identifying the weight (".weight" or ".weight_packed")
	repackWeight bool   // repack a U8 fp4 weight into U32 words

	scaleSuffix    string // required per-block / affine scale companion
	scaleRelabelU8 bool   // relabel an F8_E4M3 scale as U8 for the loader

	biasSuffix string // optional bias / zero-point companion ("" if none)

	globalSuffix     string // optional global-scale companion ("" if none)
	globalReciprocal bool   // store the global scale as its reciprocal

	ignoreSuffixes []string // companions consumed but not written (e.g. activation scales)

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
		name:             "compressed-tensors-nvfp4",
		weightSuffix:     ".weight_packed",
		repackWeight:     true,
		scaleSuffix:      ".weight_scale",
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
		repackWeight:   true,
		scaleSuffix:    ".weight_scale",
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
//
// Some MLX-converted MoE checkpoints store the MoE gate and up projections as
// separately-stacked gate_proj and up_proj tensors. These are concatenated into
// a single input_linear tensor along axis 1 at import time so the inference
// path always uses the fused layout.
func planPrequantized(inv Inventory) ([]BlobSpec, error) {
	type entry struct {
		spec     BlobSpec
		consumed []string
	}
	byWeight := make(map[string]entry)

	for _, name := range sortedTensorNames(inv) {
		// Fused gate/up takes priority over the generic path so that the
		// split gate_proj and up_proj weights are replaced by one fused blob.
		if spec, consumed, ok, err := matchSplitMLXGateUp(name, inv); err != nil {
			return nil, err
		} else if ok {
			byWeight[name] = entry{spec: spec, consumed: consumed}
			continue
		}
		if spec, consumed, ok := matchPrequant(name, inv); ok {
			byWeight[name] = entry{spec: spec, consumed: consumed}
		}
	}

	allConsumed := make(map[string]bool)
	for _, e := range byWeight {
		for _, s := range e.consumed {
			allConsumed[s] = true
		}
	}

	specs := make([]BlobSpec, 0, len(inv.Tensors))
	for _, name := range sortedTensorNames(inv) {
		if allConsumed[name] {
			continue
		}
		if e, ok := byWeight[name]; ok {
			specs = append(specs, e.spec)
			continue
		}
		t := inv.Tensors[name]
		specs = append(specs, BlobSpec{Name: name, Tensors: []TensorSpec{{Name: name, Sources: []SourceTensor{t}}}})
	}
	return specs, nil
}

// matchSplitMLXGateUp detects an MLX-converted MoE checkpoint where the gate
// and up expert projections are stored as separate
// "...switch_mlp.gate_proj.weight" and "...switch_mlp.up_proj.weight" tensors
// (each shape [E, I, H]) and returns a single BlobSpec for
// "...input_linear.weight" (shape [E, 2*I, H]) that concatenates them along
// axis 1. Scale and optional bias companions are fused identically.
//
// Returns ok=false when name is not a gate_proj weight with the expected
// companions, and returns an error when the shapes are inconsistent.
func matchSplitMLXGateUp(name string, inv Inventory) (spec BlobSpec, consumed []string, ok bool, err error) {
	mlxPattern := prequantPatterns[0] // the "mlx" pattern
	base, hasSuffix := strings.CutSuffix(name, mlxPattern.weightSuffix)
	if !hasSuffix || !strings.HasSuffix(base, ".switch_mlp.gate_proj") {
		return BlobSpec{}, nil, false, nil
	}

	gateSrc := base + mlxPattern.scaleSuffix
	if !inv.Has(gateSrc) {
		return BlobSpec{}, nil, false, nil
	}

	layerBase := strings.TrimSuffix(base, ".switch_mlp.gate_proj")
	upBase := layerBase + ".switch_mlp.up_proj"
	upWeightSrc := upBase + mlxPattern.weightSuffix
	upScaleSrc := upBase + mlxPattern.scaleSuffix
	if !inv.Has(upWeightSrc) || !inv.Has(upScaleSrc) {
		return BlobSpec{}, nil, false, nil
	}

	gateW := inv.Tensors[name]
	upW := inv.Tensors[upWeightSrc]
	if len(gateW.Shape) != 3 {
		return BlobSpec{}, nil, false, fmt.Errorf("gate_proj weight %s has unexpected shape %v (expected 3-D)", name, gateW.Shape)
	}
	if len(upW.Shape) != 3 {
		return BlobSpec{}, nil, false, fmt.Errorf("up_proj weight %s has unexpected shape %v (expected 3-D)", upWeightSrc, upW.Shape)
	}
	E, I, H := gateW.Shape[0], gateW.Shape[1], gateW.Shape[2]
	if upW.Shape[0] != E || upW.Shape[1] != I || upW.Shape[2] != H {
		return BlobSpec{}, nil, false, fmt.Errorf("gate_proj shape %v and up_proj shape %v do not match", gateW.Shape, upW.Shape)
	}
	fusedShape := []int32{E, 2 * I, H}

	outWeight := layerBase + ".input_linear.weight"
	md := prequantMetadata(inv, mlxPattern)
	consumed = append(consumed, upWeightSrc, gateSrc, upScaleSrc)

	gateScale := inv.Tensors[gateSrc]
	upScale := inv.Tensors[upScaleSrc]
	var fusedScaleShape []int32
	if len(gateScale.Shape) == 3 && gateScale.Shape[0] == E && gateScale.Shape[1] == I {
		if upScale.Shape[0] == E && upScale.Shape[1] == I && upScale.Shape[2] == gateScale.Shape[2] {
			fusedScaleShape = []int32{E, 2 * I, gateScale.Shape[2]}
		}
	}

	tensors := []TensorSpec{
		{
			Name:      outWeight,
			Sources:   []SourceTensor{gateW, upW},
			Transform: TransformConcatAxis1,
			OutShape:  fusedShape,
		},
	}

	scaleTensor := TensorSpec{
		Name:    outWeight + ".scale",
		Sources: []SourceTensor{gateScale, upScale},
	}
	if fusedScaleShape != nil {
		scaleTensor.Transform = TransformConcatAxis1
		scaleTensor.OutShape = fusedScaleShape
	}
	tensors = append(tensors, scaleTensor)

	if mlxPattern.biasSuffix != "" {
		gateBiasSrc := base + mlxPattern.biasSuffix
		upBiasSrc := upBase + mlxPattern.biasSuffix
		if inv.Has(gateBiasSrc) && inv.Has(upBiasSrc) {
			consumed = append(consumed, gateBiasSrc, upBiasSrc)
			gateB := inv.Tensors[gateBiasSrc]
			upB := inv.Tensors[upBiasSrc]
			biasTensor := TensorSpec{
				Name:    outWeight + ".bias",
				Sources: []SourceTensor{gateB, upB},
			}
			if len(gateB.Shape) == 3 && gateB.Shape[0] == E && gateB.Shape[1] == I &&
				upB.Shape[0] == E && upB.Shape[1] == I && upB.Shape[2] == gateB.Shape[2] {
				biasTensor.Transform = TransformConcatAxis1
				biasTensor.OutShape = []int32{E, 2 * I, gateB.Shape[2]}
			}
			tensors = append(tensors, biasTensor)
		}
	}

	return BlobSpec{Name: outWeight, Tensors: tensors, Metadata: md}, consumed, true, nil
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

		return BlobSpec{Name: outWeight, Tensors: tensors, Metadata: prequantMetadata(inv, p)}, consumed, true
	}
	return BlobSpec{}, nil, false
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
