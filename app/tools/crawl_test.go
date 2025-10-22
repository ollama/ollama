package tools

import (
	"context"
	"strings"
	"testing"
)

func TestGetWebpage_Name(t *testing.T) {
	tool := &BrowserCrawler{}
	if name := tool.Name(); name != "get_webpage" {
		t.Errorf("Expected name 'get_webpage', got %s", name)
	}
}

func TestGetWebpage_Description(t *testing.T) {
	tool := &BrowserCrawler{}
	desc := tool.Description()
	if desc == "" {
		t.Error("Description should not be empty")
	}
}

func TestGetWebpage_Schema(t *testing.T) {
	tool := &BrowserCrawler{}
	schema := tool.Schema()
	if schema == nil {
		t.Error("Schema should not be nil")
	}

	// Check if schema has required properties
	if schema["type"] != "object" {
		t.Error("Schema type should be 'object'")
	}

	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Error("Schema should have properties")
	}

	// Check if urls property exists
	if _, ok := properties["urls"]; !ok {
		t.Error("Schema should have 'urls' property")
	}

	// Check if required field exists
	required, ok := schema["required"].([]any)
	if !ok {
		t.Error("Schema should have 'required' field")
	}

	// Check if urls is in required
	foundUrls := false
	for _, req := range required {
		if req == "urls" {
			foundUrls = true
			break
		}
	}
	if !foundUrls {
		t.Error("'urls' should be in required fields")
	}
}

func TestGetWebpage_Execute_InvalidInput(t *testing.T) {
	tool := &BrowserCrawler{}
	ctx := context.Background()

	tests := []struct {
		name        string
		input       map[string]any
		errContains string
	}{
		{
			name:        "missing urls",
			input:       map[string]any{},
			errContains: "urls parameter is required",
		},
		{
			name: "empty urls array",
			input: map[string]any{
				"urls": []any{},
			},
			errContains: "at least one URL is required",
		},
		{
			name: "invalid urls type",
			input: map[string]any{
				"urls": "not an array",
			},
			errContains: "urls parameter is required and must be an array",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tool.Execute(ctx, tt.input)
			if err == nil {
				t.Error("Expected error but got none")
			} else if !strings.Contains(err.Error(), tt.errContains) {
				t.Errorf("Expected error containing '%s', got '%s'", tt.errContains, err.Error())
			}
		})
	}
}
