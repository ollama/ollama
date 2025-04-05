package convert

import (
	"io"

	"github.com/ollama/ollama/fs/ggml"
)

type llama4Model struct {
	TextModel   llamaModel `json:"text_config"`
	VisionModel struct{}   `json:"vision_config"`
}

// KV implements ModelConverter.
func (l *llama4Model) KV(*Tokenizer) ggml.KV {
	panic("unimplemented")
}

// Replacements implements ModelConverter.
func (l *llama4Model) Replacements() []string {
	return append(
		l.TextModel.Replacements(),
		"language_model.model", "model",
		"vision_model.model", "v",
	)
}

// Tensors implements ModelConverter.
func (l *llama4Model) Tensors([]Tensor) []ggml.Tensor {
	panic("unimplemented")
}

// specialTokenTypes implements ModelConverter.
func (l *llama4Model) specialTokenTypes() []string {
	return l.TextModel.specialTokenTypes()
}

// writeFile implements ModelConverter.
func (l *llama4Model) writeFile(io.WriteSeeker, ggml.KV, []ggml.Tensor) error {
	panic("unimplemented")
}

var _ ModelConverter = (*llama4Model)(nil)
