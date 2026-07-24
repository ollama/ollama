package create

import (
	"fmt"
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

	// TransformRelabelU8 relabels an F8_E4M3 scale as U8 so the loader reads
	// its raw bytes; the bytes themselves are unchanged.
	TransformRelabelU8 Transform = "relabel_u8"

	// TransformScalarF32 validates that the source is a scalar F32 and copies
	// it through (a global scale stored as-is).
	TransformScalarF32 Transform = "scalar_f32"

	// TransformReciprocalF32 validates a scalar F32 and stores its reciprocal
	// (a global scale the producer stored inverted).
	TransformReciprocalF32 Transform = "reciprocal_f32"

	// TransformStackExperts concatenates N per-expert source tensors (in
	// expert-index order) into one [experts, ...] tensor.
	TransformStackExperts Transform = "stack_experts"

	// TransformDecodeFP8 dequantizes a block-FP8 weight using its block scale.
	// Its two sources are the F8_E4M3 weight and its scale companion; the
	// result is a BF16 tensor, which Quantize (if set) then re-quantizes.
	TransformDecodeFP8 Transform = "decode_fp8"

	// TransformDecodeStackFP8 stacks N per-expert block-FP8 weights (and their N
	// block scales) into one [experts, out, in] tensor and dequantizes it. Its
	// sources are the N weights followed by the N scales, in expert-index order;
	// the result is a BF16 tensor, which Quantize (if set) then re-quantizes.
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
	var (
		specs []BlobSpec
		err   error
	)
	switch class.Kind {
	case SourceFloat:
		specs, err = planFloat(inv, class.Quantize, policy)
	case SourcePrequantized:
		specs, err = planPrequantized(inv)
	case SourceBlockFP8:
		specs, err = planBlockFP8(inv, class.Quantize, policy)
	default:
		return nil, fmt.Errorf("plan: source kind %q is not yet supported", class.Kind)
	}
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

// planFloat plans a float model: per-expert tensors are packed into one blob
// per layer's expert group; every other tensor becomes its own blob, with the
// quantization policy deciding which weights are quantized and to what.
func planFloat(inv Inventory, quantize string, policy quantizePolicy) ([]BlobSpec, error) {
	groups := make(map[string][]SourceTensor)
	var plain []string
	for _, name := range sortedTensorNames(inv) {
		if gp, ok := perExpertGroup(name); ok {
			groups[gp] = append(groups[gp], inv.Tensors[name])
		} else {
			plain = append(plain, name)
		}
	}

	specs := make([]BlobSpec, 0, len(plain)+len(groups))
	for _, name := range plain {
		t := inv.Tensors[name]
		q := ""
		if quantize != "" {
			q = policy.quantizationType(name, t.Shape, quantize)
		}
		specs = append(specs, BlobSpec{
			Name:    name,
			Tensors: []TensorSpec{{Name: name, Sources: []SourceTensor{t}, Quantize: q}},
		})
	}

	for _, gp := range sortedKeys(groups) {
		spec, err := planExpertGroup(gp, groups[gp], quantize, policy)
		if err != nil {
			return nil, err
		}
		specs = append(specs, spec)
	}
	return specs, nil
}

// planExpertGroup packs a layer's per-expert weights into one blob: the experts
// of each projection are stacked into a single [experts, out, in] tensor and
// quantized per the policy. Output tensor names keep the source's ".experts."
// path; only the per-expert index is dropped.
func planExpertGroup(groupPrefix string, tensors []SourceTensor, quantize string, policy quantizePolicy) (BlobSpec, error) {
	type expert struct {
		idx int
		t   SourceTensor
	}
	byProj := make(map[string][]expert)
	for _, t := range tensors {
		idx, proj, err := parseExpertTensor(groupPrefix, t.Name)
		if err != nil {
			return BlobSpec{}, err
		}
		byProj[proj] = append(byProj[proj], expert{idx: idx, t: t})
	}

	spec := BlobSpec{Name: groupPrefix}
	for _, proj := range sortedKeys(byProj) {
		experts := byProj[proj]
		sort.Slice(experts, func(i, j int) bool { return experts[i].idx < experts[j].idx })

		base := experts[0].t
		sources := make([]SourceTensor, len(experts))
		for i, e := range experts {
			if e.t.Dtype != base.Dtype || !slices.Equal(e.t.Shape, base.Shape) {
				return BlobSpec{}, fmt.Errorf("expert group %s projection %s has mismatched expert layout (%s %v vs %s %v)",
					groupPrefix, proj, base.Dtype, base.Shape, e.t.Dtype, e.t.Shape)
			}
			sources[i] = e.t
		}

		stackedName := groupPrefix + "." + proj + ".weight"
		stackedShape := append([]int32{int32(len(experts))}, base.Shape...)
		q := ""
		if quantize != "" {
			q = policy.quantizationType(stackedName, stackedShape, quantize)
		}
		spec.Tensors = append(spec.Tensors, TensorSpec{
			Name:      stackedName,
			Sources:   sources,
			Transform: TransformStackExperts,
			Quantize:  q,
			OutDtype:  base.Dtype,
			OutShape:  stackedShape,
		})
	}
	return spec, nil
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
	return idx, proj, nil
}

// perExpertGroup reports whether name is a per-expert weight that must be
// stacked — e.g. "<layer>.mlp.experts.3.gate_proj.weight" — and returns its
// group prefix. An already-stacked expert tensor (one tensor covering all
// experts, e.g. "<layer>.mlp.experts.gate_up_proj.weight" as qwen3.5 and
// gemma4 ship it) is not per-expert: it is quantized as an ordinary 3D tensor
// and the runtime splits/uses it directly.
func perExpertGroup(name string) (string, bool) {
	gp := ExpertGroupPrefix(name)
	if gp == "" {
		return "", false
	}
	rest, ok := strings.CutPrefix(name, gp+".")
	if !ok {
		return "", false
	}
	idx, _, ok := strings.Cut(rest, ".")
	if !ok {
		return "", false
	}
	if _, err := strconv.Atoi(idx); err != nil {
		return "", false
	}
	return gp, true
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
