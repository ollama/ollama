package server

import (
	"fmt"
	"strings"

	"github.com/ollama/ollama/internal/modelref"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

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
	if ref, ok, err := parseDigestModelRef(parsed); ok || err != nil {
		return ref, err
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

func parseDeleteModelRef(raw string) (parsedModelRef, error) {
	parsedRef, err := modelref.ParseRef(raw)
	if err != nil {
		return parsedModelRef{}, err
	}
	if ref, ok, err := parseDigestModelRef(parsedRef); ok || err != nil {
		return ref, err
	}

	return parseNormalizePullModelRef(raw)
}

func parseDigestModelRef(parsed modelref.ParsedRef) (parsedModelRef, bool, error) {
	name, ok, err := digestModelName(parsed.Base)
	if !ok || err != nil {
		return parsedModelRef{}, ok, err
	}
	if parsed.Source == modelSourceCloud {
		return parsedModelRef{}, true, fmt.Errorf("digest references are local: %s", parsed.Original)
	}

	return parsedModelRef{
		Original: parsed.Original,
		Base:     name.DisplayShortest(),
		Name:     name,
		Source:   parsed.Source,
	}, true, nil
}

func digestModelName(ref string) (model.Name, bool, error) {
	if runner, digestRef, ok := strings.Cut(ref, ":"); ok {
		if normalized, err := normalizeRunner(runner); err == nil && normalized != "" {
			digest, ok, err := manifest.ResolveDigestReference(digestRef)
			if !ok || err != nil {
				return model.Name{}, ok, err
			}
			name, err := runnerDigestModelName(digest, normalized)
			return name, true, err
		}
	}

	digest, ok, err := manifest.ResolveDigestReference(ref)
	if !ok || err != nil {
		return model.Name{}, ok, err
	}
	return digestReferenceName(digest), true, nil
}

func runnerDigestModelName(digest, runner string) (model.Name, error) {
	name := digestReferenceName(digest)
	m, err := manifest.ParseNamedManifestForRunner(name, runner)
	if err != nil {
		return model.Name{}, err
	}
	if selected := m.SelectedDigest(); selected != "" && !sameDigest(digest, selected) {
		name = digestReferenceName(selected)
	}
	return name, nil
}

func digestReferenceName(digest string) model.Name {
	digest = strings.ToLower(strings.Replace(digest, ":", "-", 1))
	return model.ParseName(digest)
}

func sameDigest(a, b string) bool {
	a = strings.ToLower(strings.Replace(a, "-", ":", 1))
	b = strings.ToLower(strings.Replace(b, "-", ":", 1))
	if !strings.HasPrefix(a, "sha256:") {
		a = "sha256:" + a
	}
	if !strings.HasPrefix(b, "sha256:") {
		b = "sha256:" + b
	}
	return a == b
}
