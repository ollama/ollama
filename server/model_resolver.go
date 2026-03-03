package server

import (
	"github.com/ollama/ollama/internal/modelref"
	"github.com/ollama/ollama/types/model"
)

type modelSourceRequirement = modelref.ModelSource

const (
	modelSourceUnspecified modelSourceRequirement = modelref.ModelSourceUnspecified
	modelSourceLocal       modelSourceRequirement = modelref.ModelSourceLocal
	modelSourceCloud       modelSourceRequirement = modelref.ModelSourceCloud
)

var (
	errConflictingModelSource = modelref.ErrConflictingSourceSuffix
	errModelRequired          = modelref.ErrModelRequired
)

type parsedModelRef struct {
	Original string
	Base     string
	Name     model.Name
	Source   modelSourceRequirement
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

func parseAndValidatePullModelRef(raw string) (parsedModelRef, error) {
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
