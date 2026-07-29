package server

import (
	"cmp"
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// listModels builds /api/tags from the manifests and the per-blob metadata
// files, extracting for any blob that has none yet. Manifest lists contribute
// one row per child runner.
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

		rows, err := describeModelRows(name, mf)
		if err != nil {
			slog.Warn("failed to describe model", "model", name.String(), "error", err)
			continue
		}
		models = append(models, rows...)
	}

	sortListModelResponses(models)
	return models, nil
}

// describeModelRows describes one named manifest for /api/tags. A manifest
// list describes one row per child runner, keyed by the child digest and
// carrying the parent's modification time; every other manifest describes
// itself.
func describeModelRows(name model.Name, mf *manifest.Manifest) ([]api.ListModelResponse, error) {
	parent, modified, ok, err := readManifestList(name)
	if err != nil {
		return nil, err
	}
	if !ok {
		runner, err := displayRunnerForManifest(mf)
		if err != nil {
			return nil, err
		}

		m, err := GetModel(name.String())
		if err != nil {
			slog.Warn("could not load model to describe it", "model", name.String(), "error", err)
			m = nil
		}

		row, err := describeModelFromManifest(name, mf, runner, m)
		if err != nil {
			return nil, err
		}
		return []api.ListModelResponse{row}, nil
	}

	rows := make([]api.ListModelResponse, 0, len(parent.Manifests))
	for _, child := range parent.Manifests {
		digest, err := manifest.ChildManifestDigest(child)
		if err != nil {
			return nil, err
		}
		resolved, ok, err := resolveLocalShowManifestChild(child)
		if err != nil {
			return nil, err
		}
		if !ok {
			continue
		}

		runner, err := normalizeRunner(child.Runner)
		if err != nil {
			return nil, err
		}

		m, err := GetModelForRunner(name.String(), child.Runner)
		if err != nil {
			slog.Warn("could not load model to describe it", "model", name.String(), "runner", child.Runner, "error", err)
			m = nil
		}

		row, err := describeModelFromManifest(name, resolved, runner, m)
		if err != nil {
			return nil, err
		}
		row.Digest = strings.TrimPrefix(digest, "sha256:")
		row.Size = resolved.Size()
		if !modified.IsZero() {
			row.ModifiedAt = modified
		}
		rows = append(rows, row)
	}

	return rows, nil
}

// readManifestList returns the named manifest parsed as a manifest list along
// with its modification time. It reports ok=false when the name is not a
// manifest list.
func readManifestList(name model.Name) (*manifest.Manifest, time.Time, bool, error) {
	data, err := manifest.ReadManifestData(name)
	if err != nil {
		return nil, time.Time{}, false, err
	}

	var parent manifest.Manifest
	if err := json.Unmarshal(data, &parent); err != nil {
		return nil, time.Time{}, false, err
	}
	if parent.MediaType != manifest.MediaTypeManifestList {
		return nil, time.Time{}, false, nil
	}

	path, err := manifest.ResolvePathForName(name)
	if err != nil {
		return nil, time.Time{}, false, err
	}
	fi, err := os.Lstat(path)
	if err != nil {
		return nil, time.Time{}, false, err
	}

	return &parent, fi.ModTime(), true, nil
}

// displayRunnerForManifest returns the runner /api/tags reports for a
// manifest. Legacy manifests predate runner metadata; those report the
// default the scheduler applies for the config's weight format.
func displayRunnerForManifest(mf *manifest.Manifest) (string, error) {
	runner := mf.Runner
	if runner == "" {
		cfg, err := readModelListConfig(mf)
		if err != nil {
			return "", err
		}
		runner, _ = manifest.MetadataForConfig(cfg)
	}
	if runner != "" {
		return normalizeRunner(runner)
	}
	return "", nil
}

// describeModelFromManifest describes one manifest, enriching the row from
// the loaded model when one is available. Capabilities come from the same
// Model.Capabilities() the inference path uses, so the two cannot drift.
func describeModelFromManifest(name model.Name, mf *manifest.Manifest, runner string, m *Model) (api.ListModelResponse, error) {
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
		Digest:      strings.TrimPrefix(mf.Digest(), "sha256:"),
		ModifiedAt:  modified,
		Details: api.ModelDetails{
			Format:            cfg.ModelFormat,
			Family:            cfg.ModelFamily,
			Families:          append([]string(nil), cfg.ModelFamilies...),
			ParameterSize:     cfg.ModelType,
			QuantizationLevel: cfg.FileType,
			ContextLength:     cfg.ContextLen,
			EmbeddingLength:   cfg.EmbedLen,
			Runner:            runner,
		},
	}

	// A model that will not load is the one a user most needs to see, in
	// order to remove it. Report what the manifest says.
	if m == nil {
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
			fileType := gguf.FileType(m.metadata.Int("general.file_type")).String()
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
		if c := j.ModifiedAt.Compare(i.ModifiedAt); c != 0 {
			return c
		}
		// Rows that share an mtime (manifest-list children under one parent,
		// models created within the same instant) would otherwise follow map
		// iteration order; tie-break so row order is deterministic.
		if c := cmp.Compare(i.Name, j.Name); c != 0 {
			return c
		}
		return cmp.Compare(i.Digest, j.Digest)
	})
}
