// Package decisionqwen routes Qwen3.5 decision checkpoints by their
// publisher-defined scoring format, since config.json names only the backbone.
package decisionqwen

import (
	"fmt"

	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/d3"
	"github.com/ollama/ollama/mlxrunner/model/pplxdecider"
)

func init() { model.Register("Qwen3_5Model", newModel) }

func newModel(root *model.Root) (model.Model, error) {
	var cfg struct {
		FormatID string `json:"format_id"`
	}
	if err := root.Manifest.ReadConfigJSON("decision_config.json", &cfg); err != nil {
		return nil, fmt.Errorf("read Qwen decision format: %w", err)
	}
	switch cfg.FormatID {
	case "d3-code-readout-v1":
		return d3.NewModel(root)
	case "":
		return pplxdecider.NewModel(root)
	default:
		return nil, fmt.Errorf("unsupported Qwen decision format %q", cfg.FormatID)
	}
}
