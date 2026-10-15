package create

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Laya ships an encoder config and a decision config rather than a root HF
// config. Keep both verbatim and add only the descriptor Ollama uses to dispatch.
func readLayaConfig(dir string) (sourceModelConfig, json.RawMessage, error) {
	var decision struct {
		Encoder    string `json:"encoder"`
		MaxLen     int    `json:"max_len"`
		HeadLayers *int   `json:"head_layers"`
	}
	data, err := os.ReadFile(filepath.Join(dir, "rl_agent_config.json"))
	if err != nil {
		return sourceModelConfig{}, nil, err
	}
	if err := json.Unmarshal(data, &decision); err != nil {
		return sourceModelConfig{}, nil, fmt.Errorf("parse rl_agent_config.json: %w", err)
	}
	var encoder struct {
		ModelType         string `json:"model_type"`
		HiddenSize        int    `json:"hidden_size"`
		NumHiddenLayers   int    `json:"num_hidden_layers"`
		NumAttentionHeads int    `json:"num_attention_heads"`
		VocabSize         int    `json:"vocab_size"`
	}
	data, err = os.ReadFile(filepath.Join(dir, "encoder", "config.json"))
	if err != nil {
		return sourceModelConfig{}, nil, err
	}
	if err := json.Unmarshal(data, &encoder); err != nil {
		return sourceModelConfig{}, nil, fmt.Errorf("parse encoder/config.json: %w", err)
	}
	if decision.Encoder == "" || decision.MaxLen <= 0 || decision.HeadLayers == nil || encoder.ModelType != "modernbert" {
		return sourceModelConfig{}, nil, fmt.Errorf("unsupported Laya source configuration")
	}
	cfg := sourceModelConfig{ModelType: "laya", Architectures: []string{"LayaForDecision"}}
	data, err = json.Marshal(map[string]any{
		"architectures": cfg.Architectures, "model_type": cfg.ModelType,
		"hidden_size": encoder.HiddenSize, "num_hidden_layers": encoder.NumHiddenLayers,
		"num_attention_heads": encoder.NumAttentionHeads, "vocab_size": encoder.VocabSize,
		"max_position_embeddings": decision.MaxLen,
	})
	return cfg, data, err
}
