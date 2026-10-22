package create

import (
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

// Packaging notices emitted during create: one line each, sent through the
// create status channel so both the server API and the local CLI flows print
// them without extra plumbing. Informational lines follow the established
// "using autodetected ..." style; warnings start with "warning:" and end with
// the corrective action. Notices never change exit codes or block a create.

// Template-note suffixes for AutoRendererParserNotice, naming the source
// template that becomes inert once a Go renderer/parser is resolved.
const (
	// GGUFTemplateBypassed notes that llama-server runs with jinja disabled
	// when a renderer/parser is set, so the GGUF's embedded chat template is
	// inert at runtime.
	GGUFTemplateBypassed = "the embedded chat template will not be used"

	// tokenizerTemplateBypassed notes that a safetensors source's
	// tokenizer_config.json/chat_template.jinja never renders once a Go
	// renderer is resolved.
	tokenizerTemplateBypassed = "the tokenizer's chat template will not be used"
)

// AutoRendererParserNotice reports renderer/parser names inferred from model
// metadata rather than named in the Modelfile. Pass only the inferred values;
// leave a value empty when the create request named it explicitly.
// templateNote, when non-empty, names the source template left inert.
func AutoRendererParserNotice(arch, renderer, parser, templateNote string) string {
	var what string
	switch {
	case renderer != "" && parser != "" && renderer == parser:
		what = fmt.Sprintf("renderer/parser %q", renderer)
	case renderer != "" && parser != "":
		what = fmt.Sprintf("renderer %q and parser %q", renderer, parser)
	case renderer != "":
		what = fmt.Sprintf("renderer %q", renderer)
	case parser != "":
		what = fmt.Sprintf("parser %q", parser)
	default:
		return ""
	}
	notice := fmt.Sprintf("using autodetected %s for architecture %q", what, arch)
	if templateNote != "" {
		notice += "; " + templateNote
	}
	return notice
}

// AutoStopTokensNotice reports stop tokens injected because the architecture
// requires them and the Modelfile set none.
func AutoStopTokensNotice(stops []string) string {
	if len(stops) == 0 {
		return ""
	}
	quoted := make([]string, len(stops))
	for i, stop := range stops {
		quoted[i] = fmt.Sprintf("%q", stop)
	}
	return fmt.Sprintf("using autodetected stop token %s", strings.Join(quoted, ", "))
}

// samplerDefault names a sampler parameter that will fall back to the given
// built-in default at runtime.
type samplerDefault struct {
	Name    string
	Default any
}

// MultipleModelGGUFNotice reports a create that resolved to more than one
// logical model GGUF: the extras cannot load as a second model, and a draft
// companion only wires up through an explicit DRAFT directive.
func MultipleModelGGUFNotice(names []string) string {
	if len(names) < 2 {
		return ""
	}
	return fmt.Sprintf("warning: multiple model GGUFs in one create (%s); extras are unusable (set DRAFT <file> for a draft model)", strings.Join(names, ", "))
}

// missingSamplerDefaultNotice reports one sampler parameter that neither the
// Modelfile nor the source model's defaults set, so the built-in default
// governs at runtime.
func missingSamplerDefaultNotice(name string, def any) string {
	return fmt.Sprintf("warning: %s unset; using default %s=%v (set PARAMETER %s N to change)", name, name, def, name)
}

// embeddedDraftNotice reports an embedded draft (MTP) head in the model file
// that cannot engage because draft_num_predict is unset: the server zeroes it
// unless a distinct draft model or an explicit positive value exists.
func embeddedDraftNotice() string {
	return "warning: drafting model detected but draft_num_predict unset (set PARAMETER draft_num_predict N to enable)"
}

// distinctDraftNotice reports that a packaged draft model is wired up with
// the built-in proposal depth because draft_num_predict is unset.
func distinctDraftNotice(depth int) string {
	return fmt.Sprintf("warning: draft_num_predict unset; using default draft_num_predict=%d (set PARAMETER draft_num_predict N to change)", depth)
}

// MissingRequiresNotice reports the minimum Ollama version floor the package
// will record: the create flow's default when the Modelfile names no REQUIRES.
func MissingRequiresNotice(floor string) string {
	return fmt.Sprintf("warning: no REQUIRES; using default %s (set REQUIRES to change)", floor)
}

// SamplerDefaultsNotice appends one line per temperature/top_p entry that is
// unset in both the create request parameters and the source model's
// generation defaults. The named defaults are resolved from api.DefaultOptions
// so the message matches what the server applies.
func SamplerDefaultsNotice(parameters map[string]any, generationDefaults map[string]any) []string {
	builtin := []samplerDefault{
		{Name: "temperature", Default: api.DefaultOptions().Temperature},
		{Name: "top_p", Default: api.DefaultOptions().TopP},
	}
	var notices []string
	for _, m := range builtin {
		if _, ok := parameters[m.Name]; ok {
			continue
		}
		if _, ok := generationDefaults[m.Name]; ok {
			continue
		}
		notices = append(notices, missingSamplerDefaultNotice(m.Name, m.Default))
	}
	return notices
}

// DraftNotices appends notices for draft wiring: an embedded draft head with
// nothing to enable it, or a distinct draft model relying on the built-in
// proposal depth. A distinct draft takes precedence over an embedded head at
// runtime, so the embedded notice is skipped when one is packaged.
func DraftNotices(parameters map[string]any, embeddedDraft, distinctDraft bool) []string {
	_, depthSet := parameters["draft_num_predict"]
	switch {
	case distinctDraft && !depthSet:
		return []string{distinctDraftNotice(api.DefaultOptions().Runner.DraftNumPredict)}
	case embeddedDraft && !distinctDraft && !depthSet:
		return []string{embeddedDraftNotice()}
	default:
		return nil
	}
}

// safetensorsCreateNotices assembles the packaging notices for a safetensors
// import from the resolved config, the inference provenance, and the create
// options. It owns detection shared by the local and server-side import paths.
func safetensorsCreateNotices(inv Inventory, cfg model.ConfigV2, inf configInference, opts PipelineOptions) []string {
	var notices []string

	renderer, parser := "", ""
	if inf.renderer {
		renderer = cfg.Renderer
	}
	if inf.parser {
		parser = cfg.Parser
	}
	if renderer != "" || parser != "" {
		note := ""
		if inf.hasChatTemplate {
			note = tokenizerTemplateBypassed
		}
		notices = append(notices, AutoRendererParserNotice(inv.Config.Architecture(), renderer, parser, note))
	}

	notices = append(notices, SamplerDefaultsNotice(opts.Parameters, cfg.GenerationDefaults)...)
	notices = append(notices, DraftNotices(opts.Parameters, hasEmbeddedDraft(inv), opts.DraftDir != "")...)

	if opts.Requires == "" {
		notices = append(notices, MissingRequiresNotice(SafetensorsMinOllamaVersion))
	}
	return notices
}

// hasEmbeddedDraft reports whether the source ships an embedded draft (MTP)
// head: either the config declares nextn prediction layers or mtp.* tensors
// (optionally under a container prefix) are present. Presence alone does not
// enable drafting; the notice tells the user how to enable it.
func hasEmbeddedDraft(inv Inventory) bool {
	if inv.Config.NumNextNPredictLayers > 0 {
		return true
	}
	for name := range inv.Tensors {
		if strings.HasPrefix(name, "mtp.") || strings.Contains(name, ".mtp.") {
			return true
		}
	}
	return false
}
