package server

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"text/template/parse"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/convert"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

var intermediateBlobs map[string]string = make(map[string]string)

type layerGGML struct {
	Layer
	*llm.GGML
}

func parseFromModel(ctx context.Context, name model.Name, fn func(api.ProgressResponse)) (layers []*layerGGML, err error) {
	m, err := ParseNamedManifest(name)
	switch {
	case errors.Is(err, os.ErrNotExist):
		if err := PullModel(ctx, name.String(), &RegistryOptions{}, fn); err != nil {
			return nil, err
		}

		m, err = ParseNamedManifest(name)
		if err != nil {
			return nil, err
		}
	case err != nil:
		return nil, err
	}

	for _, layer := range m.Layers {
		layer, err := NewLayerFromLayer(layer.Digest, layer.MediaType, name.DisplayShortest())
		if err != nil {
			return nil, err
		}

		switch layer.MediaType {
		case "application/vnd.ollama.image.model",
			"application/vnd.ollama.image.projector",
			"application/vnd.ollama.image.adapter":
			blobpath, err := GetBlobsPath(layer.Digest)
			if err != nil {
				return nil, err
			}

			blob, err := os.Open(blobpath)
			if err != nil {
				return nil, err
			}
			defer blob.Close()

			ggml, _, err := llm.DecodeGGML(blob, 0)
			if err != nil {
				return nil, err
			}

			layers = append(layers, &layerGGML{layer, ggml})
		default:
			layers = append(layers, &layerGGML{layer, nil})
		}
	}

	return layers, nil
}

func parseFromZipFile(_ context.Context, command string, baseLayers []*layerGGML, f *os.File, digest string, fn func(api.ProgressResponse)) (layers []*layerGGML, err error) {
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}

	r, err := zip.NewReader(f, fi.Size())
	if err != nil {
		return nil, err
	}

	p, err := os.MkdirTemp(filepath.Dir(f.Name()), "")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(p)

	fn(api.ProgressResponse{Status: "converting model"})
	// TODO(mxyng): this should write directly into a layer
	// e.g. NewLayer(arch.Reader(), "application/vnd.ollama.image.model")
	t, err := os.CreateTemp(p, "fp16")
	if err != nil {
		return nil, err
	}
	defer t.Close()
	defer os.Remove(t.Name())

	var layerType string

	switch command {
	case "adapter":
		var baseModel *llm.GGML
		for _, l := range baseLayers {
			if l.GGML != nil {
				baseModel = l.GGML
				break
			}
		}

		if baseModel == nil {
			return nil, fmt.Errorf("no base model specified for the adapter")
		}

		if err := convert.ConvertAdapter(convert.NewZipReader(r, p, 32<<20), t, baseModel.KV()); err != nil {
			return nil, err
		}
		layerType = "application/vnd.ollama.image.adapter"
	case "model":
		if err := convert.ConvertModel(convert.NewZipReader(r, p, 32<<20), t); err != nil {
			return nil, err
		}
		layerType = "application/vnd.ollama.image.model"
	}

	if _, err := t.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}

	layer, err := NewLayer(t, layerType)
	if err != nil {
		return nil, err
	}

	bin, err := layer.Open()
	if err != nil {
		return nil, err
	}
	defer bin.Close()

	ggml, _, err := llm.DecodeGGML(bin, 0)
	if err != nil {
		return nil, err
	}

	layers = append(layers, &layerGGML{layer, ggml})

	intermediateBlobs[digest] = layer.Digest
	return detectChatTemplate(layers)
}

func parseFromFile(ctx context.Context, command string, baseLayers []*layerGGML, file *os.File, digest string, fn func(api.ProgressResponse)) (layers []*layerGGML, err error) {
	sr := io.NewSectionReader(file, 0, 512)
	contentType, err := detectContentType(sr)
	if err != nil {
		return nil, err
	}

	switch contentType {
	case "gguf", "ggla":
		// noop
	case "application/zip":
		return parseFromZipFile(ctx, command, baseLayers, file, digest, fn)
	default:
		return nil, fmt.Errorf("unsupported content type: %s", contentType)
	}

	stat, err := file.Stat()
	if err != nil {
		return nil, err
	}

	var offset int64
	for offset < stat.Size() {
		ggml, n, err := llm.DecodeGGML(file, 0)
		if errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return nil, err
		}

		mediatype := "application/vnd.ollama.image.model"
		if ggml.Name() == "ggla" || ggml.KV().Kind() == "adapter" {
			mediatype = "application/vnd.ollama.image.adapter"
		} else if ggml.KV().Architecture() == "clip" {
			mediatype = "application/vnd.ollama.image.projector"
		}

		var layer Layer
		if digest != "" && n == stat.Size() && offset == 0 {
			layer, err = NewLayerFromLayer(digest, mediatype, file.Name())
			if err != nil {
				slog.Debug("could not create new layer from layer", "error", err)
			}
		}

		// Fallback to creating layer from file copy (either NewLayerFromLayer failed, or digest empty/n != stat.Size())
		if layer.Digest == "" {
			layer, err = NewLayer(io.NewSectionReader(file, offset, n), mediatype)
			if err != nil {
				return nil, err
			}
		}

		layers = append(layers, &layerGGML{layer, ggml})
		offset = n
	}

	return detectChatTemplate(layers)
}

func detectChatTemplate(layers []*layerGGML) ([]*layerGGML, error) {
	for _, layer := range layers {
		if s := layer.GGML.KV().ChatTemplate(); s != "" {
			if t, err := template.Named(s); err != nil {
				slog.Debug("template detection", "error", err)
			} else {
				layer, err := NewLayer(t.Reader(), "application/vnd.ollama.image.template")
				if err != nil {
					return nil, err
				}

				layer.status = fmt.Sprintf("using autodetected template %s", t.Name)
				layers = append(layers, &layerGGML{layer, nil})

				if t.Parameters != nil {
					var b bytes.Buffer
					if err := json.NewEncoder(&b).Encode(t.Parameters); err != nil {
						return nil, err
					}

					layer, err := NewLayer(&b, "application/vnd.ollama.image.params")
					if err != nil {
						return nil, err
					}

					layers = append(layers, &layerGGML{layer, nil})
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

	if contentType := llm.DetectGGMLType(b.Bytes()); contentType != "" {
		return contentType, nil
	}

	if contentType := http.DetectContentType(b.Bytes()); contentType != "application/octet-stream" {
		return contentType, nil
	}

	return "unknown", nil
}

func parseObjects(s string) []map[string]any {
	var objs []map[string]any
	for offset := 0; offset < len(s); {
		var obj map[string]any
		decoder := json.NewDecoder(strings.NewReader(s[offset:]))
		if err := decoder.Decode(&obj); errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			break
		} else if syntax := &(json.SyntaxError{}); errors.As(err, &syntax) {
			// skip over any syntax errors
			offset += int(syntax.Offset)
		} else if unmarshalType := &(json.UnmarshalTypeError{}); errors.As(err, &unmarshalType) {
			// skip over any unmarshalable types
			offset += int(unmarshalType.Offset)
		} else if err != nil {
			return nil
		} else {
			offset += int(decoder.InputOffset())
			objs = append(objs, obj)
		}
	}

	return objs
}

// parseToolCalls attempts to parse a JSON string into a slice of ToolCalls.
// mxyng: this only really works if the input contains tool calls in some JSON format
func (m *Model) parseToolCalls(s string) ([]api.ToolCall, bool) {
	// create a subtree from the node that ranges over .ToolCalls
	tmpl := m.Template.Subtree(func(n parse.Node) bool {
		if t, ok := n.(*parse.RangeNode); ok {
			return slices.Contains(template.Identifiers(t.Pipe), "ToolCalls")
		}

		return false
	})

	if tmpl == nil {
		return nil, false
	}

	var b bytes.Buffer
	if err := tmpl.Execute(&b, map[string][]api.ToolCall{
		"ToolCalls": {
			{
				Function: api.ToolCallFunction{
					Name: "@@name@@",
					Arguments: api.ToolCallFunctionArguments{
						"@@argument@@": 1,
					},
				},
			},
		},
	}); err != nil {
		return nil, false
	}

	templateObjects := parseObjects(b.String())
	if len(templateObjects) == 0 {
		return nil, false
	}

	// find the keys that correspond to the name and arguments fields
	var name, arguments string
	for k, v := range templateObjects[0] {
		switch v.(type) {
		case string:
			name = k
		case map[string]any:
			arguments = k
		}
	}

	if name == "" || arguments == "" {
		return nil, false
	}

	responseObjects := parseObjects(s)
	if len(responseObjects) == 0 {
		return nil, false
	}

	// collect all nested objects
	var collect func(any) []map[string]any
	collect = func(obj any) (all []map[string]any) {
		switch o := obj.(type) {
		case map[string]any:
			all = append(all, o)
			for _, v := range o {
				all = append(all, collect(v)...)
			}
		case []any:
			for _, v := range o {
				all = append(all, collect(v)...)
			}
		}

		return all
	}

	var objs []map[string]any
	for _, p := range responseObjects {
		objs = append(objs, collect(p)...)
	}

	var toolCalls []api.ToolCall
	for _, kv := range objs {
		n, nok := kv[name].(string)
		a, aok := kv[arguments].(map[string]any)
		if nok && aok {
			toolCalls = append(toolCalls, api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      n,
					Arguments: a,
				},
			})
		}
	}

	return toolCalls, len(toolCalls) > 0
}
