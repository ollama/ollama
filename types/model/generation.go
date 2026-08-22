package model

import "encoding/json"

// GenerationDefaults contains model-authored sampler defaults keyed by Ollama
// option names.
type GenerationDefaults map[string]any

type generationDefaultKind int

const (
	generationDefaultInt generationDefaultKind = iota
	generationDefaultFloat
)

type generationDefaultMapping struct {
	option   string
	hfKeys   []string
	ggufKeys []string
	kind     generationDefaultKind
}

func generationDefault(option string, kind generationDefaultKind, ggufKey string, hfKeys ...string) generationDefaultMapping {
	return generationDefaultMapping{
		option:   option,
		hfKeys:   hfKeys,
		ggufKeys: []string{ggufKey},
		kind:     kind,
	}
}

var generationDefaultMappings = []generationDefaultMapping{
	generationDefault("top_k", generationDefaultInt, "general.sampling.top_k", "top_k"),
	generationDefault("top_p", generationDefaultFloat, "general.sampling.top_p", "top_p"),
	generationDefault("min_p", generationDefaultFloat, "general.sampling.min_p", "min_p"),
	generationDefault("typical_p", generationDefaultFloat, "general.sampling.typ_p", "typical_p"),
	generationDefault("temperature", generationDefaultFloat, "general.sampling.temp", "temperature"),
	generationDefault("repeat_last_n", generationDefaultInt, "general.sampling.penalty_last_n", "repeat_last_n", "penalty_last_n"),
	generationDefault("repeat_penalty", generationDefaultFloat, "general.sampling.penalty_repeat", "repetition_penalty", "repeat_penalty", "penalty_repeat"),
	generationDefault("presence_penalty", generationDefaultFloat, "general.sampling.penalty_present", "presence_penalty"),
	generationDefault("frequency_penalty", generationDefaultFloat, "general.sampling.penalty_freq", "frequency_penalty"),
}

// GenerationDefaultOptions returns the Ollama option names that can be populated
// from model-authored generation defaults.
func GenerationDefaultOptions() []string {
	options := make([]string, 0, len(generationDefaultMappings))
	for _, mapping := range generationDefaultMappings {
		options = append(options, mapping.option)
	}

	return options
}

// ParseHFGenerationDefaults extracts sampler defaults from Hugging Face
// generation_config.json data.
func ParseHFGenerationDefaults(data []byte) (GenerationDefaults, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}

	defaults := GenerationDefaults{}
	for _, mapping := range generationDefaultMappings {
		for _, key := range mapping.hfKeys {
			b, ok := raw[key]
			if !ok {
				continue
			}

			switch mapping.kind {
			case generationDefaultInt:
				if value, ok := intGenerationDefault(b); ok {
					defaults[mapping.option] = value
				}
			case generationDefaultFloat:
				if value, ok := floatGenerationDefault(b); ok {
					defaults[mapping.option] = value
				}
			}

			if _, ok := defaults[mapping.option]; ok {
				break
			}
		}
	}

	if len(defaults) == 0 {
		return nil, nil
	}

	return defaults, nil
}

// ParseGGUFGenerationDefaults extracts sampler defaults from GGUF metadata.
func ParseGGUFGenerationDefaults(intValue func(string) (int64, bool), floatValue func(string) (float64, bool)) GenerationDefaults {
	defaults := GenerationDefaults{}
	for _, mapping := range generationDefaultMappings {
		for _, key := range mapping.ggufKeys {
			switch mapping.kind {
			case generationDefaultInt:
				if value, ok := intValue(key); ok {
					defaults[mapping.option] = value
				}
			case generationDefaultFloat:
				if value, ok := floatValue(key); ok {
					defaults[mapping.option] = value
				}
			}

			if _, ok := defaults[mapping.option]; ok {
				break
			}
		}
	}

	if len(defaults) == 0 {
		return nil
	}

	return defaults
}

func intGenerationDefault(data json.RawMessage) (int64, bool) {
	var value int64
	if err := json.Unmarshal(data, &value); err == nil {
		return value, true
	}

	var f float64
	if err := json.Unmarshal(data, &f); err != nil {
		return 0, false
	}

	// Match api.Options.FromMap; rounding may be better for near-integers.
	return int64(f), true
}

func floatGenerationDefault(data json.RawMessage) (float64, bool) {
	var value float64
	if err := json.Unmarshal(data, &value); err != nil {
		return 0, false
	}

	return value, true
}
