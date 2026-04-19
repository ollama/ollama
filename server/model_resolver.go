package server

import (
	"log/slog"
	"strings"

	"github.com/ollama/ollama/internal/modelref"
	"github.com/ollama/ollama/types/model"
)

// Temporary redirection logic to map incompatible library models to compatible versions.
//
// Architectures listed here are handled via republished blobs under the
// dhiltgen/ namespace. Once llama/compat/ grows a handler for an arch, its
// entry should be removed from this list — the compat layer translates the
// original library/ blob in memory so no republish is needed.
var compatModelRedirects = []struct{ from, to string }{
	{"library/gpt-oss", "dhiltgen/gpt-oss"},
	// library/gemma3 — handled by llama/compat (text + vision).
	{"library/embeddinggemma", "dhiltgen/embeddinggemma"},
	{"library/snowflake-arctic-embed2", "dhiltgen/snowflake-arctic-embed2"},
	{"library/gemma3n", "dhiltgen/gemma3n"},
	{"library/glm-4.7-flash", "dhiltgen/glm-4.7-flash"},
	{"library/deepseek-ocr", "dhiltgen/deepseek-ocr"},
	{"library/glm-ocr", "dhiltgen/glm-ocr"},
	{"library/gemma4", "dhiltgen/gemma4"},
	{"library/qwen2.5vl", "dhiltgen/qwen2.5vl"},
	{"library/qwen3-vl", "dhiltgen/qwen3-vl"},
}

// applyCompatRedirect checks if a model name matches a compat redirect and
// returns the redirected name. Returns the original name if no redirect applies.
func applyCompatRedirect(n model.Name) (model.Name, bool) {
	for _, r := range compatModelRedirects {
		fromNS, fromModel, _ := strings.Cut(r.from, "/")
		if fromNS == n.Namespace && fromModel == n.Model {
			redirected := n
			toNS, toRest, _ := strings.Cut(r.to, "/")
			redirected.Namespace = toNS
			// Support "namespace/model:tag" to override the tag
			if toModel, toTag, hasTag := strings.Cut(toRest, ":"); hasTag {
				redirected.Model = toModel
				redirected.Tag = toTag
			} else {
				redirected.Model = toRest
			}
			slog.Debug("redirecting to compatible model", "from", n.DisplayShortest(), "to", redirected.DisplayShortest())
			return redirected, true
		}
	}
	return n, false
}

// reverseCompatRedirect maps a redirected name back to its original library name.
// Used by PsHandler so users see the name they requested, not the internal redirect target.
// TODO: consider removing this before merging — it papers over the fact that
// the scheduler stores the redirected name instead of the user-facing name.
func reverseCompatRedirect(n model.Name) model.Name {
	for _, r := range compatModelRedirects {
		toNS, toModel, _ := strings.Cut(r.to, "/")
		if toNS == n.Namespace && toModel == n.Model {
			fromNS, fromModel, _ := strings.Cut(r.from, "/")
			reversed := n
			reversed.Namespace = fromNS
			reversed.Model = fromModel
			return reversed
		}
	}
	return n
}

type modelSource = modelref.ModelSource

const (
	modelSourceUnspecified modelSource = modelref.ModelSourceUnspecified
	modelSourceLocal       modelSource = modelref.ModelSourceLocal
	modelSourceCloud       modelSource = modelref.ModelSourceCloud
)

var (
	errConflictingModelSource = modelref.ErrConflictingSourceSuffix
	errModelRequired          = modelref.ErrModelRequired
)

type parsedModelRef struct {
	// Original is the caller-provided model string before source parsing.
	// Example: "gpt-oss:20b:cloud".
	Original string
	// Base is the model string after source suffix normalization.
	// Example: "gpt-oss:20b:cloud" -> "gpt-oss:20b".
	Base string
	// Name is Base parsed as a fully-qualified model.Name with defaults applied.
	// Example: "registry.ollama.ai/library/gpt-oss:20b".
	Name model.Name
	// Source captures explicit source intent from the original input.
	// Example: "gpt-oss:20b:cloud" -> modelSourceCloud.
	Source modelSource
}

func parseAndValidateModelRef(raw string) (parsedModelRef, error) {
	var zero parsedModelRef

	parsed, err := modelref.ParseRef(raw)
	if err != nil {
		return zero, err
	}

	name := model.ParseName(parsed.Base)
	if !name.IsValid() {
		return zero, model.Unqualified(name)
	}

	return parsedModelRef{
		Original: parsed.Original,
		Base:     parsed.Base,
		Name:     name,
		Source:   parsed.Source,
	}, nil
}

func parseNormalizePullModelRef(raw string) (parsedModelRef, error) {
	var zero parsedModelRef

	parsedRef, err := modelref.ParseRef(raw)
	if err != nil {
		return zero, err
	}

	normalizedName, _, err := modelref.NormalizePullName(raw)
	if err != nil {
		return zero, err
	}

	name := model.ParseName(normalizedName)
	if !name.IsValid() {
		return zero, model.Unqualified(name)
	}

	return parsedModelRef{
		Original: parsedRef.Original,
		Base:     normalizedName,
		Name:     name,
		Source:   parsedRef.Source,
	}, nil
}
