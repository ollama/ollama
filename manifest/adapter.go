package manifest

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"

	"github.com/ollama/ollama/types/model"
)

const MediaTypePEFTAdapter = "application/vnd.ollama.image.adapter.safetensors"
const AdapterConfigName = "adapter_config.json"
const AdapterWeightsName = "adapter_model.safetensors"

// AdapterRef pins both the factors and their scaling configuration to blobs.
type AdapterRef struct {
	Config  string `json:"config"`
	Weights string `json:"weights"`
}

func (m *Manifest) AdapterSize() int64 {
	if m != nil {
		for _, l := range m.Layers {
			if l.MediaType == MediaTypePEFTAdapter {
				return l.Size
			}
		}
	}
	return 0
}

func (m *Manifest) Adapter() (AdapterRef, error) {
	var ref AdapterRef
	for _, l := range m.Layers {
		switch {
		case l.MediaType == MediaTypeImageJSON && l.Name == AdapterConfigName:
			if ref.Config != "" {
				return ref, fmt.Errorf("duplicate adapter config")
			}
			ref.Config = l.Digest
		case l.MediaType == MediaTypePEFTAdapter:
			if ref.Weights != "" {
				return ref, fmt.Errorf("multiple PEFT adapters are unsupported")
			}
			ref.Weights = l.Digest
		}
	}
	if (ref.Config == "") != (ref.Weights == "") {
		return ref, fmt.Errorf("incomplete PEFT adapter layers")
	}
	return ref, nil
}

// MLXBaseKey identifies the inputs to the native model loader. Tag metadata
// (prompts, parameters, provenance) and request-selected adapters can vary.
func (m *Manifest) MLXBaseKey(draft *model.Draft) string {
	h := sha256.New()
	enc := json.NewEncoder(h)
	_ = enc.Encode(draft)
	for _, l := range m.Layers {
		if l.MediaType == MediaTypeImageTensor || l.MediaType == MediaTypeImageJSON && l.Name != AdapterConfigName {
			_ = enc.Encode([3]string{l.MediaType, l.Name, l.Digest})
		}
	}
	return fmt.Sprintf("mlx:%x", h.Sum(nil))
}
