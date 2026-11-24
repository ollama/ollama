package renderers

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestNemotron3NanoRenderer_Images(t *testing.T) {
	tests := []struct {
		name     string
		msgs     []api.Message
		expected string
	}{
		{
			name: "single image inserts placeholder",
			msgs: []api.Message{
				{Role: "user", Content: "Describe this image.", Images: []api.ImageData{api.ImageData("img1")}},
			},
			expected: "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n[img-0] Describe this image.<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "generic image placeholder is rewritten",
			msgs: []api.Message{
				{Role: "user", Content: "[img]Describe this image.", Images: []api.ImageData{api.ImageData("img1")}},
			},
			expected: "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n[img-0]Describe this image.<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "image offsets increment across messages",
			msgs: []api.Message{
				{Role: "user", Content: "Describe the first image.", Images: []api.ImageData{api.ImageData("img1")}},
				{Role: "assistant", Content: "It shows something."},
				{Role: "user", Content: "Compare these.", Images: []api.ImageData{api.ImageData("img2"), api.ImageData("img3")}},
			},
			expected: "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n[img-0] Describe the first image.<|im_end|>\n<|im_start|>assistant\n<think></think>It shows something.<|im_end|>\n<|im_start|>user\n[img-1][img-2] Compare these.<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			renderer := &Nemotron3NanoRenderer{}
			rendered, err := renderer.Render(tt.msgs, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(tt.expected, rendered); diff != "" {
				t.Fatalf("mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestNemotron3NanoRendererSchemaKeywords(t *testing.T) {
	var tool api.Tool
	if err := json.Unmarshal([]byte(`{"type":"function","function":{"name":"find","parameters":{"type":"object","properties":{
		"year":{"type":"integer","default":2022,"minimum":1900},
		"strict":{"type":"boolean","default":false},
		"tag":{"type":"string","default":null},
		"venue":{"type":"object","properties":{"city":{"type":"string","default":"Toronto"}},"additionalProperties":false}
	},"additionalProperties":false,"description":"Search filters."}}}`), &tool); err != nil {
		t.Fatal(err)
	}

	rendered, err := (&Nemotron3NanoRenderer{}).Render([]api.Message{{Role: "user", Content: "go"}}, []api.Tool{tool}, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := "<parameter>\n<name>year</name>\n<type>integer</type>\n<default>2022</default>\n<minimum>1900</minimum>\n</parameter>\n" +
		"<parameter>\n<name>strict</name>\n<type>boolean</type>\n<default>False</default>\n</parameter>\n" +
		"<parameter>\n<name>tag</name>\n<type>string</type>\n<default>None</default>\n</parameter>\n" +
		"<parameter>\n<name>venue</name>\n<type>object</type>\n<properties>{\"city\": {\"type\": \"string\", \"default\": \"Toronto\"}}</properties>\n<additionalProperties>False</additionalProperties>\n</parameter>\n" +
		"<additionalProperties>False</additionalProperties>\n<description>Search filters.</description>\n</parameters>"
	if !strings.Contains(rendered, want) {
		t.Fatalf("rendered tool block mismatch, got:\n%s", rendered)
	}
}
