package create

import "fmt"

// CreateDraftLayers imports a draft (speculative-decoding / MTP assistant)
// safetensors model into prefixed tensor and config blobs and returns the
// layers WITHOUT writing a manifest — the caller folds them into the target
// model's manifest. A draft never stands alone; it always accompanies a target
// model named on the Modelfile's FROM line.
//
// It runs the same read → classify → plan → write pipeline as Create. Output
// tensor names keep their source form, namespaced by tensorPrefix (e.g.
// "draft.") so they cannot collide with the target's tensors; config blobs are
// named under configPrefix (e.g. "draft/").
func CreateDraftLayers(modelDir, tensorPrefix, configPrefix, quantize string, store BlobStore, fn func(status string)) ([]LayerInfo, error) {
	if tensorPrefix == "" {
		return nil, fmt.Errorf("draft tensor prefix must not be empty")
	}
	if configPrefix == "" {
		return nil, fmt.Errorf("draft config prefix must not be empty")
	}
	defer sweepMLX()

	inv, err := ReadInventory(modelDir)
	if err != nil {
		return nil, fmt.Errorf("read draft model: %w", err)
	}
	class, err := Classify(inv, quantize)
	if err != nil {
		return nil, err
	}
	policy, err := newTensorImportTransform(inv)
	if err != nil {
		return nil, fmt.Errorf("build draft quantization policy for %q: %w", inv.Config.Architecture(), err)
	}
	specs, err := Plan(inv, class, draftPolicy{policy})
	if err != nil {
		return nil, fmt.Errorf("plan draft model: %w", err)
	}
	specs = prefixSpecs(specs, tensorPrefix)

	fn(fmt.Sprintf("importing draft (%d tensors%s)", len(inv.Tensors), quantizeStatus(class)))
	layers, err := WriteBlobs(specs, modelDir, store)
	if err != nil {
		return nil, err
	}

	configLayers, _, err := importConfigBlobs(modelDir, configPrefix, store, fn)
	if err != nil {
		return nil, err
	}
	return append(layers, configLayers...), nil
}

// prefixSpecs returns specs with prefix prepended to every output blob name and
// output tensor name, leaving the source references (which point at the source
// files) untouched. Scale/bias keys derive from the tensor name, so they inherit
// the prefix automatically.
func prefixSpecs(specs []BlobSpec, prefix string) []BlobSpec {
	out := make([]BlobSpec, len(specs))
	for i, spec := range specs {
		tensors := make([]TensorSpec, len(spec.Tensors))
		for j, ts := range spec.Tensors {
			ts.Name = prefix + ts.Name
			tensors[j] = ts
		}
		out[i] = BlobSpec{Name: prefix + spec.Name, Tensors: tensors, Metadata: spec.Metadata}
	}
	return out
}

// draftPolicy wraps an architecture policy to keep a draft model's token
// embedding at source precision (drafts start with unquantized embeddings; this
// may change later). Every other tensor follows the wrapped policy. It is given
// unprefixed source names, since planning runs before prefixSpecs.
type draftPolicy struct{ inner quantizePolicy }

func (p draftPolicy) quantizationType(name string, shape []int32, requested string) string {
	if isEmbedTokensWeight(name) {
		return ""
	}
	return p.inner.quantizationType(name, shape, requested)
}
