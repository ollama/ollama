package server

import (
	"cmp"
	"context"
	"encoding/json"
	"log/slog"
	"slices"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

type modelListSummary struct {
	Model        string
	Name         string
	RemoteModel  string
	RemoteHost   string
	Size         int64
	Digest       string
	ModifiedAt   time.Time
	Details      api.ModelDetails
	Capabilities []model.Capability
}

// listModels builds /api/tags from the manifests and the per-blob metadata
// metadata files. A model whose file is missing has it extracted here, so the first
// list after an upgrade pays for the models it has not seen before and later
// ones read only small files.
func listModels(ctx context.Context) ([]api.ListModelResponse, error) {
	manifests, err := manifest.Manifests(true)
	if err != nil {
		return nil, err
	}

	models := make([]api.ListModelResponse, 0, len(manifests))
	for name, mf := range manifests {
		if ctx != nil {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}

		summary, err := buildModelListSummary(name, mf)
		if err != nil {
			slog.Warn("failed to describe model", "model", name.String(), "error", err)
			continue
		}
		models = append(models, summary.ListModelResponse())
	}

	sortListModelResponses(models)
	return models, nil
}

// buildModelListSummary describes one model for /api/tags. Capabilities come
// from the same Model.Capabilities() the inference path uses, so the two cannot
// drift; the rest is manifest and config bookkeeping.
func buildModelListSummary(name model.Name, mf *manifest.Manifest) (modelListSummary, error) {
	cfg, err := readModelListConfig(mf)
	if err != nil {
		return modelListSummary{}, err
	}

	var modified time.Time
	if fi := mf.FileInfo(); fi != nil {
		modified = fi.ModTime()
	}

	summary := modelListSummary{
		Model:       name.DisplayShortest(),
		Name:        name.DisplayShortest(),
		RemoteModel: cfg.RemoteModel,
		RemoteHost:  cfg.RemoteHost,
		Size:        mf.Size(),
		Digest:      mf.Digest(),
		ModifiedAt:  modified,
		Details: api.ModelDetails{
			Format:            cfg.ModelFormat,
			Family:            cfg.ModelFamily,
			Families:          append([]string(nil), cfg.ModelFamilies...),
			ParameterSize:     cfg.ModelType,
			QuantizationLevel: cfg.FileType,
			ContextLength:     cfg.ContextLen,
			EmbeddingLength:   cfg.EmbedLen,
		},
	}

	m, err := GetModel(name.String())
	if err != nil {
		return modelListSummary{}, err
	}
	summary.Details.ParentModel = m.ParentModel
	summary.Capabilities = m.Capabilities()

	if m.ModelPath != "" && m.isGGUF() {
		if summary.Details.ContextLength == 0 {
			summary.Details.ContextLength = int(m.metadata.Int("context_length"))
		}
		if summary.Details.EmbeddingLength == 0 {
			summary.Details.EmbeddingLength = int(m.metadata.Int("embedding_length"))
		}
		if m.metadata.Valid("general.file_type") {
			fileType := ggml.FileType(m.metadata.Int("general.file_type")).String()
			if isUnknownQuantization(summary.Details.QuantizationLevel) && !isUnknownQuantization(fileType) {
				summary.Details.QuantizationLevel = fileType
			}
		}
	}

	return summary, nil
}

func readModelListConfig(mf *manifest.Manifest) (model.ConfigV2, error) {
	var cfg model.ConfigV2
	if mf == nil || mf.Config.Digest == "" {
		return cfg, nil
	}

	f, err := mf.Config.Open()
	if err != nil {
		return cfg, err
	}
	defer f.Close()

	if err := json.NewDecoder(f).Decode(&cfg); err != nil {
		return cfg, err
	}

	return cfg, nil
}

func isUnknownQuantization(quantization string) bool {
	return quantization == "" || quantization == "unknown"
}

func (s modelListSummary) ListModelResponse() api.ListModelResponse {
	resp := api.ListModelResponse{
		Model:       s.Model,
		Name:        s.Name,
		RemoteModel: s.RemoteModel,
		RemoteHost:  s.RemoteHost,
		Size:        s.Size,
		Digest:      s.Digest,
		ModifiedAt:  s.ModifiedAt,
		Details: api.ModelDetails{
			ParentModel:       s.Details.ParentModel,
			Format:            s.Details.Format,
			Family:            s.Details.Family,
			Families:          append([]string(nil), s.Details.Families...),
			ParameterSize:     s.Details.ParameterSize,
			QuantizationLevel: s.Details.QuantizationLevel,
			ContextLength:     s.Details.ContextLength,
			EmbeddingLength:   s.Details.EmbeddingLength,
		},
	}

	resp.Capabilities = append([]model.Capability(nil), s.Capabilities...)

	return resp
}

func sortListModelResponses(models []api.ListModelResponse) {
	slices.SortStableFunc(models, func(i, j api.ListModelResponse) int {
		// Preserve the existing /api/tags order: most recently modified first.
		return cmp.Compare(j.ModifiedAt.Unix(), i.ModifiedAt.Unix())
	})
}
