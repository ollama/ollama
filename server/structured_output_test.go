package server

import (
	"encoding/json"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestMLXSinglePassFormatUsesThinkingBoundary(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"colour":{"type":"string"}},"required":["colour"]}`)
	prompt := "<|im_start|>assistant\n<think>\n"
	mlx := &Model{Config: model.ConfigV2{ModelFormat: "safetensors", Parser: "qwen3.5"}}

	got := mlxSinglePassFormat(mlx, schema, prompt, "", "", &api.ThinkValue{Value: true})
	if len(got) == 0 {
		t.Fatal("expected a structural tag for MLX thinking output")
	}

	var envelope struct {
		Type   string `json:"type"`
		Format struct {
			Type     string `json:"type"`
			Elements []struct {
				Type       string          `json:"type"`
				Excludes   []string        `json:"excludes"`
				Value      string          `json:"value"`
				JSONSchema json.RawMessage `json:"json_schema"`
			} `json:"elements"`
		} `json:"format"`
	}
	if err := json.Unmarshal(got, &envelope); err != nil {
		t.Fatal(err)
	}
	if envelope.Type != "structural_tag" || envelope.Format.Type != "sequence" {
		t.Fatalf("got grammar envelope %#v", envelope)
	}
	if len(envelope.Format.Elements) != 3 {
		t.Fatalf("got %d grammar elements, want 3", len(envelope.Format.Elements))
	}
	if envelope.Format.Elements[0].Type != "any_text" || len(envelope.Format.Elements[0].Excludes) != 1 || envelope.Format.Elements[0].Excludes[0] != "</think>" {
		t.Fatalf("first element does not stop at the thinking boundary: %#v", envelope.Format.Elements[0])
	}
	if envelope.Format.Elements[1].Type != "const_string" || envelope.Format.Elements[1].Value != "</think>" {
		t.Fatalf("second element does not close thinking: %#v", envelope.Format.Elements[1])
	}
	if envelope.Format.Elements[2].Type != "json_schema" || string(envelope.Format.Elements[2].JSONSchema) != string(schema) {
		t.Fatalf("third element does not preserve the schema: %#v", envelope.Format.Elements[2])
	}
}

func TestMLXSinglePassFormatIncludesOpeningTagWhenPromptDoesNotPrefill(t *testing.T) {
	schema := json.RawMessage(`{"type":"object"}`)
	mlx := &Model{Config: model.ConfigV2{ModelFormat: "safetensors", Parser: "qwen3.5"}}

	got := mlxSinglePassFormat(mlx, schema, "assistant:", "", "", &api.ThinkValue{Value: true})
	var envelope map[string]any
	if err := json.Unmarshal(got, &envelope); err != nil {
		t.Fatal(err)
	}
	elements := envelope["format"].(map[string]any)["elements"].([]any)
	first := elements[0].(map[string]any)
	if first["type"] != "tag" || first["begin"] != "<think>" || first["end"] != "</think>" {
		t.Fatalf("got opening element %#v", first)
	}
}

func TestMLXSinglePassFormatKeepsExistingBehaviorWithoutThinking(t *testing.T) {
	mlx := &Model{Config: model.ConfigV2{ModelFormat: "safetensors", Parser: "qwen3.5"}}
	if got := mlxSinglePassFormat(mlx, json.RawMessage(`{"type":"object"}`), "assistant:", "", "", &api.ThinkValue{Value: false}); got != nil {
		t.Fatalf("got a single-pass grammar for think=false: %s", got)
	}
}

func TestMLXSinglePassFormatSkipsNonThinkingQwen3Parser(t *testing.T) {
	mlx := &Model{Config: model.ConfigV2{ModelFormat: "safetensors", Parser: "qwen3"}}
	if got := mlxSinglePassFormat(mlx, json.RawMessage(`{"type":"object"}`), "assistant:", "", "", &api.ThinkValue{Value: true}); got != nil {
		t.Fatalf("got a single-pass grammar for non-thinking qwen3 parser: %s", got)
	}
}
