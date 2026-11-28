package server

import (
	"encoding/json"
	"maps"

	"github.com/ollama/ollama/create"
	fsgguf "github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
)

// ggufCreateFacts carries what createModel learned while walking the base
// layers, so ggufCreateNotices can report packaging defaults without
// re-opening any blobs.
type ggufCreateFacts struct {
	arch             string
	metadata         *fsgguf.Metadata
	modelFiles       []string
	inferredRenderer string
	inferredParser   string
	autoStops        []string
	hasDraftLayer    bool
	requires         string
	requiresAuto     bool
	params           map[string]any
}

// ggufCreateNotices assembles the informational and warning lines for a local
// GGUF create. See create/notices.go for message wording.
func ggufCreateNotices(facts ggufCreateFacts) []string {
	var notices []string

	if facts.inferredRenderer != "" || facts.inferredParser != "" {
		note := ""
		if facts.metadata != nil && facts.metadata.ChatTemplate() != "" {
			note = create.GGUFTemplateBypassed
		}
		notices = append(notices, create.AutoRendererParserNotice(facts.arch, facts.inferredRenderer, facts.inferredParser, note))
	}
	if len(facts.autoStops) > 0 {
		notices = append(notices, create.AutoStopTokensNotice(facts.autoStops))
	}

	if facts.metadata != nil {
		notices = append(notices, create.SamplerDefaultsNotice(facts.params, ggufFileGenerationDefaults(facts.metadata))...)
	}
	notices = append(notices, create.DraftNotices(facts.params, facts.metadata != nil && llm.HasEmbeddedDraft(facts.metadata), facts.hasDraftLayer)...)

	if notice := create.MultipleModelGGUFNotice(facts.modelFiles); notice != "" {
		notices = append(notices, notice)
	}

	if facts.requiresAuto {
		notices = append(notices, create.MissingRequiresNotice(facts.requires))
	}
	return notices
}

// ggufFileGenerationDefaults reads model-authored sampler defaults from GGUF
// metadata, mirroring generationDefaultsFromMetadata for the *fsgguf.Metadata
// createModel already holds.
func ggufFileGenerationDefaults(md *fsgguf.Metadata) model.GenerationDefaults {
	return model.ParseGGUFGenerationDefaults(
		func(key string) (int64, bool) {
			value := md.KeyValue(key)
			if n, ok := value.IntOK(); ok {
				return n, true
			}
			if f, ok := value.FloatOK(); ok {
				return int64(f), true
			}
			return 0, false
		},
		func(key string) (float64, bool) {
			return md.KeyValue(key).FloatOK()
		},
	)
}

// createParams merges parameters inherited from base model layers with the
// create request's, in the same order ApplyModelfileLayers applies them: the
// first inherited layer wins, and request parameters override all inherited
// values. Unreadable inherited layers are skipped: a missed key is at worst a
// noisy notice.
func createParams(baseLayers []*modelLayer, requestParams map[string]any) map[string]any {
	params := make(map[string]any)
	for _, layer := range baseLayers {
		if layer.MediaType != "application/vnd.ollama.image.params" {
			continue
		}
		f, err := layer.Open()
		if err != nil {
			continue
		}
		var inherited map[string]any
		decodeErr := json.NewDecoder(f).Decode(&inherited)
		closeErr := f.Close()
		if decodeErr != nil || closeErr != nil {
			continue
		}
		for k, v := range inherited {
			if _, exists := params[k]; !exists {
				params[k] = v
			}
		}
	}
	maps.Copy(params, requestParams)
	return params
}
