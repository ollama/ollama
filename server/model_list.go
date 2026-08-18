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

// listModels builds /api/tags from the manifests and the per-blob metadata
// files, extracting for any blob that has none yet.
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

		summary, err := describeModel(name, mf)
		if err != nil {
			slog.Warn("failed to describe model", "model", name.String(), "error", err)
			continue
		}
		models = append(models, summary)
	}

	sortListModelResponses(models)
	return models, nil
}

// describeModel describes one model for /api/tags. Capabilities come from the
// same Model.Capabilities() the inference path uses, so the two cannot drift.
func describeModel(name model.Name, mf *manifest.Manifest) (api.ListModelResponse, error) {
	cfg, err := readModelListConfig(mf)
	if err != nil {
		return api.ListModelResponse{}, err
	}

	var modified time.Time
	if fi := mf.FileInfo(); fi != nil {
		modified = fi.ModTime()
	}

	summary := api.ListModelResponse{
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
		// A model that will not load is the one a user most needs to see, in
		// order to remove it. Report what the manifest says.
		slog.Warn("could not load model to describe it", "model", name.String(), "error", err)
		return summary, nil
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

func sortListModelResponses(models []api.ListModelResponse) {
	slices.SortStableFunc(models, func(i, j api.ListModelResponse) int {
		// Preserve the existing /api/tags order: most recently modified first.
		return cmp.Compare(j.ModifiedAt.Unix(), i.ModifiedAt.Unix())
	})
}
