package autotrustdecision

import (
	"encoding/json"
	"fmt"
)

type config struct {
	ModelType  string `json:"model_type"`
	MaxChoices int    `json:"max_choices"`
}

func parseConfig(data []byte, head decisionHead) (config, error) {
	var cfg config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return config{}, fmt.Errorf("parse AutoTrust decision config: %w", err)
	}
	if cfg.MaxChoices == 0 {
		choiceRange := head.Slots.Ranges["choice"]
		cfg.MaxChoices = choiceRange[1] - choiceRange[0]
		// The 27B-VL checkpoint supports wide labels despite its compact head.
		if cfg.ModelType == "qwen3_5" {
			cfg.MaxChoices = 256
		}
	}
	if cfg.MaxChoices < 2 || cfg.MaxChoices > 256 {
		return config{}, fmt.Errorf("invalid AutoTrust decision max_choices %d: must be between 2 and 256", cfg.MaxChoices)
	}
	return cfg, nil
}
