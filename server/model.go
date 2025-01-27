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
	"slices"
	"strings"
	"text/template/parse"

	"github.com/ollama/ollama/api"
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
		if err := PullModel(ctx, name.String(), &registryOptions{}, fn); err != nil {
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
	tmpl := m.Template.Sub(func(n parse.Node) bool {
		if t, ok := n.(*parse.RangeNode); ok {
			return slices.Contains(template.Identifiers(t.Pipe), "ToolCalls")
		}

		return false
	})

	if tmpl == nil {
		return nil, false
	}

	var b bytes.Buffer
	if err := tmpl.Template().Execute(&b, map[string][]api.ToolCall{
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
