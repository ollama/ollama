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
	"regexp"
	"slices"
	"strings"
	"text/template/parse"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

var intermediateBlobs map[string]string = make(map[string]string)

type layerGGML struct {
	Layer
	*ggml.GGML
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

			f, _, err := ggml.Decode(blob, 0)
			if err != nil {
				return nil, err
			}

			layers = append(layers, &layerGGML{layer, f})
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
				slog.Debug("template detection", "error", err, "template", s)
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

	if contentType := ggml.DetectContentType(b.Bytes()); contentType != "" {
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

// Get tool call token from model template
func (m *Model) TemplateToolToken() (string, string, bool) {
	// Try to detect the tool call format from the model's template
	tmpl := m.Template.Subtree(func(n parse.Node) bool {
		if t, ok := n.(*parse.RangeNode); ok {
			return slices.Contains(template.Identifiers(t.Pipe), "ToolCalls")
		}
		return false
	})

	// fmt.Println("tool call template", tmpl)
	if tmpl != nil {
		// Execute template with test data to see the format
		var b bytes.Buffer
		if err := tmpl.Execute(&b, map[string][]api.ToolCall{
			"ToolCalls": {
				{
					Function: api.ToolCallFunction{
						Name: "function_name",
						Arguments: api.ToolCallFunctionArguments{
							"argument1": "value1",
							// "argument2": "value2",
						},
					},
				},
			},
		}); err == nil {
			// Look for special tokens in the template output
			output := strings.TrimSpace(b.String())
			slog.Debug("tool call template output", "output", output)
			if strings.Contains(output, "<") {
				// Extract the special token between < and >
				start := strings.Index(output, "<")
				end := strings.Index(output, ">")
				if start >= 0 && end > start {
					token := output[start : end+1]
					return output, token, true
				}
			} else if strings.Contains(output, "[") {
				// Check if it's a tool call token rather than JSON array
				start := strings.Index(output, "[")
				end := strings.Index(output, "]")
				if start >= 0 && end > start {
					token := output[start : end+1]
					// Only consider it a token if it's not valid JSON
					var jsonTest any
					if err := json.Unmarshal([]byte(token), &jsonTest); err != nil {
						return output, token, true
					}
				}
			}
		}
	}
	return "", "", false
}

func parsePythonFunctionCall(s string) ([]api.ToolCall, bool) {
	re := regexp.MustCompile(`(\w+)\((.*?)\)`)
	matches := re.FindAllStringSubmatchIndex(s, -1)
	if len(matches) == 0 {
		return nil, false
	}

	var toolCalls []api.ToolCall
	for _, match := range matches {
		name := s[match[2]:match[3]]
		args := s[match[4]:match[5]]

		arguments := make(api.ToolCallFunctionArguments)
		if strings.Contains(args, "=") { // Keyword args
			pairs := strings.SplitSeq(args, ",")
			for pair := range pairs {
				pair = strings.TrimSpace(pair)
				kv := strings.Split(pair, "=")
				if len(kv) == 2 {
					key := strings.TrimSpace(kv[0])
					value := strings.TrimSpace(kv[1])
					arguments[key] = value
				}
			}
			toolCalls = append(toolCalls, api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      name,
					Arguments: arguments,
				},
			})
		}
	}

	if len(toolCalls) > 0 {
		return toolCalls, true
	}
	return nil, false
}

// ToolCallFormat represents different possible formats for tool calls
type toolCallFormat struct {
	// Direct format
	Name      string         `json:"name,omitempty"`
	Arguments map[string]any `json:"arguments,omitempty"`

	// Command-r-plus format
	ToolName   string         `json:"tool_name,omitempty"`
	Parameters map[string]any `json:"parameters,omitempty"`

	// Function format
	Function *struct {
		Name       string         `json:"name"`
		Arguments  map[string]any `json:"arguments,omitempty"`
		Parameters map[string]any `json:"parameters,omitempty"`
	} `json:"function,omitempty"`

	// Xlam format
	ToolCalls []toolCallFormat `json:"tool_calls,omitempty"`
}

func parseJSONToolCalls(obj map[string]any) ([]api.ToolCall, bool) {
	// Helper to convert any to []any safely
	toArray := func(v any) []any {
		if arr, ok := v.([]any); ok {
			return arr
		}
		return nil
	}

	// Convert a single format to a tool call
	makeToolCall := func(f toolCallFormat) (api.ToolCall, bool) {
		switch {
		case f.Name != "" && f.Arguments != nil:
			return api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      f.Name,
					Arguments: f.Arguments,
				},
			}, true
		case f.Name != "" && f.Parameters != nil: // Handle parameters field
			return api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      f.Name,
					Arguments: f.Parameters,
				},
			}, true
		case f.ToolName != "" && f.Parameters != nil:
			return api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      f.ToolName,
					Arguments: f.Parameters,
				},
			}, true
		case f.Function != nil && f.Function.Name != "":
			args := f.Function.Arguments
			if args == nil {
				args = f.Function.Parameters
			}
			if args != nil {
				return api.ToolCall{
					Function: api.ToolCallFunction{
						Name:      f.Function.Name,
						Arguments: args,
					},
				}, true
			}
		}
		return api.ToolCall{}, false
	}

	// Try parsing as array first
	if arr := toArray(obj); arr != nil {
		var calls []api.ToolCall
		for _, item := range arr {
			if itemMap, ok := item.(map[string]any); ok {
				var format toolCallFormat
				data, _ := json.Marshal(itemMap)
				if err := json.Unmarshal(data, &format); err == nil {
					if call, ok := makeToolCall(format); ok {
						calls = append(calls, call)
					}
				}
			}
		}
		if len(calls) > 0 {
			return calls, true
		}
	}

	// Try parsing as single object
	var format toolCallFormat
	data, _ := json.Marshal(obj)
	if err := json.Unmarshal(data, &format); err != nil {
		return nil, false
	}

	// Handle xlam format (tool_calls array)
	if len(format.ToolCalls) > 0 {
		var calls []api.ToolCall
		for _, f := range format.ToolCalls {
			if call, ok := makeToolCall(f); ok {
				calls = append(calls, call)
			}
		}
		if len(calls) > 0 {
			return calls, true
		}
	}

	// Try as single tool call
	if call, ok := makeToolCall(format); ok {
		return []api.ToolCall{call}, true
	}

	return nil, false
}

// token, partial, success
func deriveToolToken(s string, prefix string) (string, bool, bool) {
	// There shouldn't be spaces in a tool token
	if len(strings.Fields(s)) > 1 {
		return "", false, false
	}

	if prefix == "[" && len(s) > 1 && s[len(s)-1] == ']' {
		return s, false, true
	} else if prefix == "<" && len(s) > 1 && s[len(s)-1] == '>' {
		return s, false, true
	}
	return "", true, true
}

func parseJSON(s string) ([]api.ToolCall, bool) {
	objs := parseObjects(s)
	tcs := []api.ToolCall{}
	for _, obj := range objs {
		toolCalls, ok := parseJSONToolCalls(obj)
		if ok {
			tcs = append(tcs, toolCalls...)
		}
	}
	if len(tcs) > 0 {
		return tcs, true
	}
	return nil, false
}

// returns tool calls, partial, success
func (m *Model) ParseToolCalls(s string, toolToken *string) ([]api.ToolCall, bool, bool) {
	// [ case can either be JSON, Python or a Tool Token
	s = strings.TrimSpace(s)
	fmt.Printf("ParseToolCallsNew input: %q\n", s)
	if len(s) == 0 {
		return nil, false, false
	}

	if strings.HasPrefix(s, "[") {
		fmt.Println("Found [ prefix")
		// JSON case
		// we do not consider array JSONs as tool calls
		if strings.HasPrefix(s, "[{") {
			fmt.Println("Found [{ prefix - attempting JSON parse")
			// TODO: mark as JSON partial
			if calls, ok := parseJSON(s); ok {
				fmt.Printf("Successfully parsed JSON, found %d calls\n", len(calls))
				return calls, false, true
			}
			return nil, true, true
		}
		// Python Case
		// We just do a full python check here
		fmt.Println("Attempting Python function parse")
		tc, ok := parsePythonFunctionCall(s)
		if ok {
			fmt.Printf("Successfully parsed Python function: %+v\n", tc)
			return tc, false, true
		}
		// Tool Token Case - this is okay if it's a real tool token and we couldn't get from template
		fmt.Println("Attempting to derive tool token")
		if toolToken == nil || *toolToken == "" {
			toolTok, partial, ok := deriveToolToken(s, "[")
			if !ok {
				return nil, false, false
			}
			if partial {
				return nil, true, true
			}
			*toolToken = toolTok
		}
		fmt.Printf("Found tool token: %q\n", *toolToken)
		s = strings.TrimSpace(s[len(*toolToken):])
		fmt.Printf("Recursing with remaining string: %q\n", s)
		if toolCalls, partial, ok := m.ParseToolCalls(s, toolToken); ok {
			return toolCalls, partial, true
		}
		return nil, true, true
	} else if strings.HasPrefix(s, "{") || strings.HasPrefix(s, "```") {
		// // TODO: temp fix
		// if strings.HasPrefix(s, "```") && len(s) == 3 {
		// 	return nil, false, false
		// }
		fmt.Println("Found { prefix - attempting JSON parse with ", s)
		if calls, ok := parseJSON(s); ok {
			fmt.Printf("Successfully parsed JSON object, found %d calls\n", len(calls))
			return calls, false, true
		}
		fmt.Println("Failed to parse JSON in JSON case")
		// TODO: possible case where it never finishes parsing - then what?
		return nil, true, true
	} else if strings.HasPrefix(s, "<") {
		fmt.Println("Found < prefix - attempting to derive tool token")
		if toolToken == nil || *toolToken == "" {
			toolTok, partial, ok := deriveToolToken(s, "<")
			if !ok {
				return nil, false, false
			}
			if partial {
				return nil, true, true
			}
			*toolToken = toolTok
			fmt.Printf("Found tool token: %q\n", *toolToken)
		}
		fmt.Printf("Found tool token: %q\n", *toolToken)
		s = strings.TrimSpace(s[len(*toolToken):])
		fmt.Printf("Recursing with remaining string: %q\n", s)
		if toolCalls, partial, ok := m.ParseToolCalls(s, toolToken); ok {
			return toolCalls, partial, true
		}
		return nil, true, true
	} else if strings.Contains(s, "(") || len(strings.Fields(s)) == 1 {
		fmt.Println("Attempting Python function parse")
		tc, ok := parsePythonFunctionCall(s)
		if ok {
			fmt.Printf("Successfully parsed Python function: %+v\n", tc)
			return tc, false, true
		}
		fmt.Printf("Failed to parse Python function: %q, returning partial", s)
		return nil, true, true
	}
	fmt.Println("No successful parse paths found")
	fmt.Printf("failed string: %q\n", s)
	return nil, false, false
}
