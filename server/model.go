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
		slog.Debug("parseToolCalls: no ToolCalls template found")
		return nil, false
	}

	slog.Debug("parseToolCalls: executing template with test data", "input", s)

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
		slog.Debug("parseToolCalls: template execution failed", "error", err)
		return nil, false
	}

	slog.Debug("parseToolCalls: template executed successfully", "output", b.String())

	templateObjects := parseObjects(b.String())
	if len(templateObjects) == 0 {
		return nil, false
	}

	slog.Debug("parseToolCalls: template objects", "objects", templateObjects)

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

func (m *Model) GetToolCallFormat(s string) (string, string, bool) {
	// Try to detect the tool call format from the model's template
	tmpl := m.Template.Subtree(func(n parse.Node) bool {
		if t, ok := n.(*parse.RangeNode); ok {
			return slices.Contains(template.Identifiers(t.Pipe), "Content")
		}
		return false
	})

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

func parsePythonFunctionCall(s string) (api.ToolCall, bool) {
	re := regexp.MustCompile(`(\w+)\((.*?)\)`)
	if match := re.FindStringSubmatchIndex(s); match != nil {
		name := s[match[2]:match[3]]
		args := s[match[4]:match[5]]

		// Check if there's a < after the closing bracket
		if idx := strings.Index(s[match[5]:], "<"); idx >= 0 {
			// Wait for closing > by returning false
			if !strings.Contains(s[match[5]+idx:], ">") {
				return api.ToolCall{}, false
			}
		}

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
			return api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      name,
					Arguments: arguments,
				},
			}, true
		}
	}
	return api.ToolCall{}, false
}

func (m *Model) ParseToolCallsStream(s string, prefix *string, specialToken *string) ([]api.ToolCall, bool, bool) {
	// The prefix check for for the tags shouldn't really be used and we should be consuming this from the model
	// Knowing what the tool token enables quicker and more reliable parsing
	// TODO: not sure how we're going to handle chatting before the tool call
	// TODO: detection would be relying on the model to know what the tool token is
	// fmt.Println("parsing tool calls", s)

	if prefix == nil {
		prefix = new(string)
		*prefix = ""
	}
	if specialToken == nil {
		specialToken = new(string)
		*specialToken = ""
	}
	// TODO: cache this
	// _, token, ok := m.GetToolCallFormat(s)
	// if ok && token != "" {
	// 	fmt.Println("token", token)
	// 	*specialToken = token
	// }
	// fmt.Println("prefix", *prefix)
	// fmt.Println("special token", *specialToken)
	var partial bool

	s = strings.TrimSpace(s)
	if len(s) == 0 {
		return nil, false, false
	}

	if specialToken != nil && len(*specialToken) > 0 {
		s2 := *specialToken
		if strings.HasPrefix(s, string(s2[0])) {
			// fmt.Println("prefix 1 is", string(s2[0]))
			partial = true
			*prefix = string(s2[0])
		}
	}

	if len(s) > 0 {
		if s[0] == '[' {
			s = strings.ReplaceAll(s, "\n", "")
			// tool call list with no special token
			if len(s) > 1 && s[1] == '{' {
				// fmt.Println("prefix 2 in [{", string(s[0]))
				partial = true
				*specialToken = "[{"
				*prefix = "[{"
			} else if *specialToken == "" {
				// possible tool call with special token but not in template
				// split s over spaces to check for special token
				if len(s) > 0 && s[len(s)-1] == ']' {
					partial = true
					*specialToken = s
					*prefix = "["
				}
			}
		} else if s[0] == '{' {
			// fmt.Println("prefix 2 in {", string(s[0]))
			partial = true
			*specialToken = "{"
			*prefix = "{"
		} else if s[0] == '<' {
			// TODO: the only issue here is that we might miss a > if the token is weird
			// The </ works as that would only happen for a tool call mainly for when we create a tag
			if len(s) > 1 && s[1] == '/' {
				// fmt.Println("prefix3 in <", string(s[0]))
				// returning a partial here is a hack to ensure that we don't send the content downstream
				return nil, true, true
				// TODO: jank hack to get special token right
				// special token might not be set yet
			} else if s[len(s)-1] == '>' {
				partial = true
				*specialToken = s
				*prefix = "<"
			} else if specialToken != nil && *specialToken == "" {
				partial = true
				*specialToken = "<"
				*prefix = "<"
			}
		}
	}

	// fmt.Println("special token", *specialToken)
	// fmt.Println("prefix", *prefix)

	if !partial {
		return nil, false, false
	}
	// Look for <function_call> tags
	// fmt.Println("looking for special token", *specialToken)
	start := strings.Index(s, *specialToken)
	if start == -1 {
		if partial {
			// fmt.Println("did not find opening tag, partial match", *specialToken)
			return nil, true, true
		}
		return nil, false, false
	}
	end := len(s)

	// Extract content between tags
	var content string
	// fmt.Println("prefix before is", *prefix)
	if *prefix == "[{" || *prefix == "{" {
		content = s[start:end]
	} else {
		content = s[start+len(*specialToken) : end]
	}
	content = strings.TrimSpace(content)
	// fmt.Println("content", content)

	var toolCalls []api.ToolCall

	// Try parsing as JSON first - could be single object or array
	var jsonObj any
	if err := json.Unmarshal([]byte(content), &jsonObj); err == nil {
		// Try as single object
		if obj, ok := jsonObj.(map[string]any); ok {
			// fmt.Println("obj", obj)
			if calls, ok := parseJSONToolCalls(obj); ok {
				toolCalls = append(toolCalls, calls...)
			}
		}
		// Try as array of objects
		if arr, ok := jsonObj.([]any); ok {
			for _, item := range arr {
				if obj, ok := item.(map[string]any); ok {
					if calls, ok := parseJSONToolCalls(obj); ok {
						toolCalls = append(toolCalls, calls...)
					}
				}
			}
		}
	} else {
		// TODO: review this case
		// Check for partial JSON before trying Python style
		if strings.HasPrefix(content, "{") || strings.HasPrefix(content, "[{") {
			// We have an opening brace/bracket but failed to parse - likely partial JSON
			return nil, true, true
		}

		// Try parsing as Python function call
		if toolCall, ok := parsePythonFunctionCall(content); ok {
			toolCalls = append(toolCalls, toolCall)
		}
	}

	// Only return success if we found valid tool calls and no errors
	if len(toolCalls) > 0 {
		// Check if any of the tool calls are malformed
		for _, call := range toolCalls {
			if call.Function.Name == "" || len(call.Function.Arguments) == 0 {
				return nil, false, false
			}
		}
		return toolCalls, false, true
	}

	// fmt.Println("no tool calls found, partial match", partial)
	if partial {
		return nil, true, true
	}
	return nil, false, false
}
