package tools

import (
	"os"
	"path/filepath"
	"testing"
)

func TestValidateSkillSpec(t *testing.T) {
	tests := []struct {
		name    string
		spec    SkillSpec
		wantErr bool
		errMsg  string
	}{
		{
			name: "valid script skill",
			spec: SkillSpec{
				Name:        "test_skill",
				Description: "A test skill",
				Parameters: []SkillParameter{
					{Name: "input", Type: "string", Description: "Input value", Required: true},
				},
				Executor: SkillExecutor{
					Type:    "script",
					Command: "echo",
				},
			},
			wantErr: false,
		},
		{
			name: "valid http skill",
			spec: SkillSpec{
				Name:        "http_skill",
				Description: "An HTTP skill",
				Executor: SkillExecutor{
					Type: "http",
					URL:  "https://example.com/api",
				},
			},
			wantErr: false,
		},
		{
			name: "missing name",
			spec: SkillSpec{
				Description: "A skill without name",
				Executor:    SkillExecutor{Type: "script", Command: "echo"},
			},
			wantErr: true,
			errMsg:  "name is required",
		},
		{
			name: "missing description",
			spec: SkillSpec{
				Name:     "no_desc",
				Executor: SkillExecutor{Type: "script", Command: "echo"},
			},
			wantErr: true,
			errMsg:  "description is required",
		},
		{
			name: "missing executor type",
			spec: SkillSpec{
				Name:        "no_exec_type",
				Description: "Missing executor type",
				Executor:    SkillExecutor{Command: "echo"},
			},
			wantErr: true,
			errMsg:  "executor.type is required",
		},
		{
			name: "script missing command",
			spec: SkillSpec{
				Name:        "script_no_cmd",
				Description: "Script without command",
				Executor:    SkillExecutor{Type: "script"},
			},
			wantErr: true,
			errMsg:  "executor.command is required",
		},
		{
			name: "http missing url",
			spec: SkillSpec{
				Name:        "http_no_url",
				Description: "HTTP without URL",
				Executor:    SkillExecutor{Type: "http"},
			},
			wantErr: true,
			errMsg:  "executor.url is required",
		},
		{
			name: "unknown executor type",
			spec: SkillSpec{
				Name:        "unknown_type",
				Description: "Unknown executor",
				Executor:    SkillExecutor{Type: "invalid"},
			},
			wantErr: true,
			errMsg:  "unknown executor type",
		},
		{
			name: "parameter missing name",
			spec: SkillSpec{
				Name:        "param_no_name",
				Description: "Parameter without name",
				Parameters:  []SkillParameter{{Type: "string", Description: "desc"}},
				Executor:    SkillExecutor{Type: "script", Command: "echo"},
			},
			wantErr: true,
			errMsg:  "parameter name is required",
		},
		{
			name: "parameter missing type",
			spec: SkillSpec{
				Name:        "param_no_type",
				Description: "Parameter without type",
				Parameters:  []SkillParameter{{Name: "foo", Description: "desc"}},
				Executor:    SkillExecutor{Type: "script", Command: "echo"},
			},
			wantErr: true,
			errMsg:  "parameter type is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateSkillSpec(tt.spec)
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error containing %q, got nil", tt.errMsg)
				} else if tt.errMsg != "" && !contains(err.Error(), tt.errMsg) {
					t.Errorf("expected error containing %q, got %q", tt.errMsg, err.Error())
				}
			} else if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestLoadSkillsFile(t *testing.T) {
	// Create a temporary skills file
	tmpDir := t.TempDir()
	skillsPath := filepath.Join(tmpDir, "skills.json")

	content := `{
		"version": "1",
		"skills": [
			{
				"name": "echo_skill",
				"description": "Echoes the input",
				"parameters": [
					{"name": "message", "type": "string", "description": "Message to echo", "required": true}
				],
				"executor": {
					"type": "script",
					"command": "echo",
					"timeout": 30
				}
			}
		]
	}`

	if err := os.WriteFile(skillsPath, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	file, err := LoadSkillsFile(skillsPath)
	if err != nil {
		t.Fatalf("LoadSkillsFile failed: %v", err)
	}

	if file.Version != "1" {
		t.Errorf("expected version 1, got %s", file.Version)
	}

	if len(file.Skills) != 1 {
		t.Fatalf("expected 1 skill, got %d", len(file.Skills))
	}

	skill := file.Skills[0]
	if skill.Name != "echo_skill" {
		t.Errorf("expected name 'echo_skill', got %s", skill.Name)
	}
	if skill.Executor.Timeout != 30 {
		t.Errorf("expected timeout 30, got %d", skill.Executor.Timeout)
	}
	if len(skill.Parameters) != 1 {
		t.Errorf("expected 1 parameter, got %d", len(skill.Parameters))
	}
}

func TestRegisterSkillsFromFile(t *testing.T) {
	tmpDir := t.TempDir()
	skillsPath := filepath.Join(tmpDir, "skills.json")

	content := `{
		"version": "1",
		"skills": [
			{
				"name": "skill_a",
				"description": "Skill A",
				"executor": {"type": "script", "command": "echo"}
			},
			{
				"name": "skill_b",
				"description": "Skill B",
				"executor": {"type": "script", "command": "cat"}
			}
		]
	}`

	if err := os.WriteFile(skillsPath, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	registry := NewRegistry()
	if err := RegisterSkillsFromFile(registry, skillsPath); err != nil {
		t.Fatalf("RegisterSkillsFromFile failed: %v", err)
	}

	if registry.Count() != 2 {
		t.Errorf("expected 2 tools, got %d", registry.Count())
	}

	names := registry.Names()
	if names[0] != "skill_a" || names[1] != "skill_b" {
		t.Errorf("unexpected tool names: %v", names)
	}
}

func TestSkillToolSchema(t *testing.T) {
	spec := SkillSpec{
		Name:        "test_tool",
		Description: "A test tool",
		Parameters: []SkillParameter{
			{Name: "required_param", Type: "string", Description: "Required", Required: true},
			{Name: "optional_param", Type: "number", Description: "Optional", Required: false},
		},
		Executor: SkillExecutor{Type: "script", Command: "echo"},
	}

	tool := NewSkillTool(spec)

	if tool.Name() != "test_tool" {
		t.Errorf("expected name 'test_tool', got %s", tool.Name())
	}

	if tool.Description() != "A test tool" {
		t.Errorf("expected description 'A test tool', got %s", tool.Description())
	}

	schema := tool.Schema()
	if schema.Name != "test_tool" {
		t.Errorf("schema name mismatch")
	}

	if len(schema.Parameters.Required) != 1 {
		t.Errorf("expected 1 required param, got %d", len(schema.Parameters.Required))
	}
	if schema.Parameters.Required[0] != "required_param" {
		t.Errorf("wrong required param: %v", schema.Parameters.Required)
	}
}

func TestSkillToolExecuteScript(t *testing.T) {
	spec := SkillSpec{
		Name:        "echo_test",
		Description: "Echo test",
		Executor: SkillExecutor{
			Type:    "script",
			Command: "cat",
			Timeout: 5,
		},
	}

	tool := NewSkillTool(spec)

	// cat will read JSON from stdin and output it
	result, err := tool.Execute(map[string]any{"message": "hello"})
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	if !contains(result, "message") || !contains(result, "hello") {
		t.Errorf("expected JSON output with 'message' and 'hello', got: %s", result)
	}
}

func TestSkillToolExecuteHTTPNotImplemented(t *testing.T) {
	spec := SkillSpec{
		Name:        "http_test",
		Description: "HTTP test",
		Executor: SkillExecutor{
			Type: "http",
			URL:  "https://example.com",
		},
	}

	tool := NewSkillTool(spec)
	_, err := tool.Execute(map[string]any{})
	if err == nil {
		t.Error("expected error for http executor")
	}
	if !contains(err.Error(), "not yet implemented") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestLoadSkillsFileNotFound(t *testing.T) {
	_, err := LoadSkillsFile("/nonexistent/path/skills.json")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestLoadSkillsFileInvalidJSON(t *testing.T) {
	tmpDir := t.TempDir()
	skillsPath := filepath.Join(tmpDir, "invalid.json")

	if err := os.WriteFile(skillsPath, []byte("not valid json"), 0644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	_, err := LoadSkillsFile(skillsPath)
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestRegisterSkillsFromFileInvalidSkill(t *testing.T) {
	tmpDir := t.TempDir()
	skillsPath := filepath.Join(tmpDir, "invalid_skill.json")

	content := `{
		"version": "1",
		"skills": [
			{
				"name": "",
				"description": "Missing name",
				"executor": {"type": "script", "command": "echo"}
			}
		]
	}`

	if err := os.WriteFile(skillsPath, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	registry := NewRegistry()
	err := RegisterSkillsFromFile(registry, skillsPath)
	if err == nil {
		t.Error("expected error for invalid skill")
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
