package server

import (
	"fmt"
	"slices"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/types/model"
)

// Exact template identities keep legacy metadata from leaking to custom templates.
// These entries describe the existing local endpoint behavior, without migrations.
var legacyThinking = map[string]model.Thinking{
	// Qwen3 template with explicit /think and /no_think controls.
	"sha256:ae370d884f108d16e7cc8fd5259ebc5773a0afa6e078b11f4ed7e39a27e0dfc4": {Values: []any{false, true}, Default: true},
	// Older Qwen3 template always opens a thinking block.
	"sha256:2d54db2b9bb29ce7db54fea63a891f5859603813c555b1f88b5e0994652897f9": {Values: []any{true}, Default: true},
	// Llama3.2 tool-use template.
	"sha256:966de95ca8a62200913e3f8bfbf84c8494536f1b94b49166851e76644e966396": {Values: []any{false}, Default: false},
}

// Thinking returns the effective local serving contract. Remote models obtain
// their current contract from the remote show response.
func (m *Model) Thinking() *model.Thinking {
	if m == nil || m.Config.RemoteHost != "" || shouldUseHarmony(m) {
		return nil
	}
	if name := resolveRendererName(m); name != "" {
		// The default-on policy only applies when the model has the thinking
		// capability. Custom renderer/parser combinations may not have it.
		if !slices.Contains(m.Capabilities(), model.CapabilityThinking) {
			return renderers.ThinkingForRenderer(name)
		}
		return thinkingForLocalRenderer(name)
	}
	if shouldUseGoTemplate(m) {
		if thinking, ok := legacyThinking[m.templateDigest]; ok {
			return thinking.Clone()
		}
	}
	return nil
}

func thinkingForLocalRenderer(name string) *model.Thinking {
	thinking := renderers.ThinkingForRenderer(name)
	if thinking == nil {
		return nil
	}
	// Local requests historically enable thinking before rendering. Preserve that
	// endpoint default even when the renderer alone defaults off.
	if thinking.Default == false && thinking.Supports(true) {
		thinking.Default = true
	}
	// true currently reaches Qwen3.8 as medium via ThinkValue.String().
	if name == "qwen3.8" {
		thinking.Default = "medium"
	}
	return thinking
}

// genericThinking excludes Harmony and template-only paths from new fallback rules.
func (m *Model) genericThinking() *model.Thinking {
	if m == nil || m.Config.Renderer == "" {
		return nil
	}
	return m.Thinking()
}

// lookupThinking lets compatibility middleware preserve named efforts only for
// local generic renderers. Other models keep the existing protocol conversions.
func lookupThinking(name string) *model.Thinking {
	m, err := GetModel(name)
	if err != nil {
		return nil
	}
	return m.genericThinking()
}

func validateLegacyThinking(think *api.ThinkValue) error {
	if think == nil || !think.IsString() {
		return nil
	}
	switch think.String() {
	case "low", "medium", "high", "max":
		return nil
	default:
		return fmt.Errorf("invalid think value: %q (must be \"high\", \"medium\", \"low\", \"max\", true, or false)", think.String())
	}
}
