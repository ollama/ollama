package agent

import (
	"testing"
)

func TestCloudModelOptionStruct(t *testing.T) {
	// Test that the struct is defined correctly
	models := []CloudModelOption{
		{Name: "glm-4.7:cloud", Description: "GLM 4.7 Cloud"},
		{Name: "qwen3-coder:480b-cloud", Description: "Qwen3 Coder 480B"},
	}

	if len(models) != 2 {
		t.Errorf("expected 2 models, got %d", len(models))
	}

	if models[0].Name != "glm-4.7:cloud" {
		t.Errorf("expected glm-4.7:cloud, got %s", models[0].Name)
	}

	if models[1].Description != "Qwen3 Coder 480B" {
		t.Errorf("expected 'Qwen3 Coder 480B', got %s", models[1].Description)
	}
}
