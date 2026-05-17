package server

import (
	"github.com/ollama/ollama/internal/modelref"
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
