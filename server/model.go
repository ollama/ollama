package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

var intermediateBlobs map[string]string = make(map[string]string)

type layerGGML struct {
	manifest.Layer
	*ggml.GGML
	splitParts []splitGGUFPart
}

func parseFromModel(ctx context.Context, name model.Name, fn func(api.ProgressResponse)) ([]*layerGGML, model.ConfigV2, error) {
	var config model.ConfigV2
	m, err := manifest.ParseNamedManifest(name)
	switch {
	case errors.Is(err, os.ErrNotExist):
		if err := PullModel(ctx, name.String(), "", &registryOptions{}, fn); err != nil {
			return nil, config, err
		}

		m, err = manifest.ParseNamedManifest(name)
		if err != nil {
			return nil, config, err
		}
	case err != nil:
		return nil, config, err
	}

	if m.Config.Digest == "" {
		return nil, config, fmt.Errorf("model %s is missing its config", name.DisplayShortest())
	}
	configFile, err := m.Config.Open()
	if err != nil {
		return nil, config, fmt.Errorf("open config for %s: %w", name.DisplayShortest(), err)
	}
	if err := json.NewDecoder(configFile).Decode(&config); err != nil {
		configFile.Close()
		return nil, config, fmt.Errorf("decode config for %s: %w", name.DisplayShortest(), err)
	}
	if err := configFile.Close(); err != nil {
		return nil, config, fmt.Errorf("close config for %s: %w", name.DisplayShortest(), err)
	}

	var layers []*layerGGML
	for _, srcLayer := range m.Layers {
		layer, err := manifest.NewLayerFromLayer(srcLayer.Digest, srcLayer.MediaType, name.DisplayShortest())
		if err != nil {
			return nil, config, err
		}
		layer.Name = srcLayer.Name

		switch layer.MediaType {
		case "application/vnd.ollama.image.model",
			"application/vnd.ollama.image.projector",
			"application/vnd.ollama.image.adapter",
			manifest.MediaTypeImageDraft:
			blobpath, err := manifest.BlobsPath(layer.Digest)
			if err != nil {
				return nil, config, err
			}

			blob, err := os.Open(blobpath)
			if err != nil {
				return nil, config, err
			}
			defer blob.Close()

			f, err := ggml.Decode(blob, -1)
			if err != nil {
				return nil, config, err
			}

			layers = append(layers, &layerGGML{Layer: layer, GGML: f})
		default:
			layers = append(layers, &layerGGML{Layer: layer})
		}
	}

	return layers, config, nil
}

func detectChatTemplate(layers []*layerGGML) ([]*layerGGML, error) {
	for _, layer := range layers {
		if s := layer.GGML.KV().ChatTemplate(); s != "" {
			if t, err := template.Named(s); err != nil {
				slog.Debug("template detection", "error", err, "template", s)
			} else {
				layer, err := manifest.NewLayer(t.Reader(), "application/vnd.ollama.image.template")
				if err != nil {
					return nil, err
				}

				layer.Status = fmt.Sprintf("using autodetected template %s", t.Name)
				layers = append(layers, &layerGGML{Layer: layer})

				if t.Parameters != nil {
					var b bytes.Buffer
					if err := json.NewEncoder(&b).Encode(t.Parameters); err != nil {
						return nil, err
					}

					layer, err := manifest.NewLayer(&b, "application/vnd.ollama.image.params")
					if err != nil {
						return nil, err
					}

					layers = append(layers, &layerGGML{Layer: layer})
				}
			}
		}
	}

	return layers, nil
}

func detectContentType(r io.Reader) (string, error) {
	var b bytes.Buffer
	if _, err := io.Copy(&b, r); err != nil {
		return "", err
	}

	if contentType := ggml.DetectContentType(b.Bytes()); contentType != "" {
		return contentType, nil
	}

	if contentType := http.DetectContentType(b.Bytes()); contentType != "application/octet-stream" {
		return contentType, nil
	}

	return "unknown", nil
}
