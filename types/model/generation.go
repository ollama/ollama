package model

import "encoding/json"

// GenerationDefaults contains model-authored sampler defaults keyed by Ollama
// option names.
type GenerationDefaults map[string]any

// ParseHFGenerationDefaults extracts sampler defaults from Hugging Face
// generation_config.json data.
func ParseHFGenerationDefaults(data []byte) (GenerationDefaults, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}

	defaults := GenerationDefaults{}
	addInt := func(hfKey, option string) {
		if b, ok := raw[hfKey]; ok {
			var value int64
			if err := json.Unmarshal(b, &value); err == nil {
				defaults[option] = value
			}
		}
	}
	addFloat := func(hfKey, option string) {
		if b, ok := raw[hfKey]; ok {
			var value float64
			if err := json.Unmarshal(b, &value); err == nil {
				defaults[option] = value
			}
		}
	}

	addInt("top_k", "top_k")
	addFloat("top_p", "top_p")
	addFloat("min_p", "min_p")
	addFloat("temperature", "temperature")
	addFloat("repetition_penalty", "repeat_penalty")
	addFloat("penalty_repeat", "repeat_penalty")
	addInt("penalty_last_n", "repeat_last_n")

	if len(defaults) == 0 {
		return nil, nil
	}

	return defaults, nil
}
