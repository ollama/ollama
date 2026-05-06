package launch

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestOpenCodeIntegration(t *testing.T) {
	o := &OpenCode{}

	t.Run("String", func(t *testing.T) {
		if got := o.String(); got != "OpenCode" {
			t.Errorf("String() = %q, want %q", got, "OpenCode")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = o
	})

	t.Run("implements Editor", func(t *testing.T) {
		var _ Editor = o
	})
}

func TestOpenCodeEdit(t *testing.T) {
	t.Run("builds config content with provider", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		o := &OpenCode{}
		if err := o.Edit([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}

		var cfg map[string]any
		if err := json.Unmarshal([]byte(o.configContent), &cfg); err != nil {
			t.Fatalf("configContent is not valid JSON: %v", err)
		}

		// Verify provider structure
		provider, _ := cfg["provider"].(map[string]any)
		ollama, _ := provider["ollama"].(map[string]any)
		if ollama["name"] != "Ollama" {
			t.Errorf("provider name = %v, want Ollama", ollama["name"])
		}
		if ollama["npm"] != "@ai-sdk/openai-compatible" {
			t.Errorf("npm = %v, want @ai-sdk/openai-compatible", ollama["npm"])
		}

		// Verify model exists
		models, _ := ollama["models"].(map[string]any)
		if models["llama3.2"] == nil {
			t.Error("model llama3.2 not found in config content")
		}

		// Verify default model
		if cfg["model"] != "ollama/llama3.2" {
			t.Errorf("model = %v, want ollama/llama3.2", cfg["model"])
		}
	})

	t.Run("multiple models", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		o := &OpenCode{}
		if err := o.Edit([]string{"llama3.2", "qwen3:32b"}); err != nil {
			t.Fatal(err)
		}

		var cfg map[string]any
		json.Unmarshal([]byte(o.configContent), &cfg)
		provider, _ := cfg["provider"].(map[string]any)
		ollama, _ := provider["ollama"].(map[string]any)
		models, _ := ollama["models"].(map[string]any)

		if models["llama3.2"] == nil {
			t.Error("model llama3.2 not found")
		}
		if models["qwen3:32b"] == nil {
			t.Error("model qwen3:32b not found")
		}
		// First model should be the default
		if cfg["model"] != "ollama/llama3.2" {
			t.Errorf("default model = %v, want ollama/llama3.2", cfg["model"])
		}
	})

	t.Run("empty models is no-op", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		o := &OpenCode{}
		if err := o.Edit([]string{}); err != nil {
			t.Fatal(err)
		}
		if o.configContent != "" {
			t.Errorf("expected empty configContent for no models, got %s", o.configContent)
		}
	})

	t.Run("does not write config files", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		o := &OpenCode{}
		o.Edit([]string{"llama3.2"})

		configDir := filepath.Join(tmpDir, ".config", "opencode")

		if _, err := os.Stat(filepath.Join(configDir, "opencode.json")); !os.IsNotExist(err) {
			t.Error("opencode.json should not be created")
		}
		if _, err := os.Stat(filepath.Join(configDir, "opencode.jsonc")); !os.IsNotExist(err) {
			t.Error("opencode.jsonc should not be created")
		}
	})

	t.Run("cloud model has limits", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		o := &OpenCode{}
		if err := o.Edit([]string{"glm-4.7:cloud"}); err != nil {
			t.Fatal(err)
		}

		var cfg map[string]any
		json.Unmarshal([]byte(o.configContent), &cfg)
		provider, _ := cfg["provider"].(map[string]any)
		ollama, _ := provider["ollama"].(map[string]any)
		models, _ := ollama["models"].(map[string]any)
		entry, _ := models["glm-4.7:cloud"].(map[string]any)

		limit, ok := entry["limit"].(map[string]any)
		if !ok {
			t.Fatal("cloud model should have limit set")
		}
		expected := cloudModelLimits["glm-4.7"]
		if limit["context"] != float64(expected.Context) {
			t.Errorf("context = %v, want %d", limit["context"], expected.Context)
		}
		if limit["output"] != float64(expected.Output) {
			t.Errorf("output = %v, want %d", limit["output"], expected.Output)
		}
	})

	t.Run("local model has no limits", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		o := &OpenCode{}
		o.Edit([]string{"llama3.2"})

		var cfg map[string]any
		json.Unmarshal([]byte(o.configContent), &cfg)
		provider, _ := cfg["provider"].(map[string]any)
		ollama, _ := provider["ollama"].(map[string]any)
		models, _ := ollama["models"].(map[string]any)
		entry, _ := models["llama3.2"].(map[string]any)

		if entry["limit"] != nil {
			t.Errorf("local model should not have limit, got %v", entry["limit"])
		}
	})
}

func TestOpenCodeModels_ReturnsNil(t *testing.T) {
	o := &OpenCode{}
	if models := o.Models(); models != nil {
		t.Errorf("Models() = %v, want nil", models)
	}
}

func TestOpenCodePaths(t *testing.T) {
	t.Run("returns nil when model.json does not exist", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		o := &OpenCode{}
		if paths := o.Paths(); paths != nil {
			t.Errorf("Paths() = %v, want nil", paths)
		}
	})

	t.Run("returns model.json path when it exists", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		stateDir := filepath.Join(tmpDir, ".local", "state", "opencode")
		os.MkdirAll(stateDir, 0o755)
		os.WriteFile(filepath.Join(stateDir, "model.json"), []byte(`{}`), 0o644)

		o := &OpenCode{}
		paths := o.Paths()
		if len(paths) != 1 {
			t.Fatalf("Paths() returned %d paths, want 1", len(paths))
		}
		if paths[0] != filepath.Join(stateDir, "model.json") {
			t.Errorf("Paths() = %v, want %v", paths[0], filepath.Join(stateDir, "model.json"))
		}
	})
}

func TestLookupCloudModelLimit(t *testing.T) {
	tests := []struct {
		name        string
		wantOK      bool
		wantContext int
		wantOutput  int
	}{
		{"glm-4.7", false, 0, 0},
		{"glm-4.7:cloud", true, 202_752, 131_072},
		{"glm-5:cloud", true, 202_752, 131_072},
		{"glm-5.1:cloud", true, 202_752, 131_072},
		{"gemma4:31b-cloud", true, 262_144, 131_072},
		{"gpt-oss:120b-cloud", true, 131_072, 131_072},
		{"gpt-oss:20b-cloud", true, 131_072, 131_072},
		{"kimi-k2.5", false, 0, 0},
		{"kimi-k2.5:cloud", true, 262_144, 262_144},
		{"deepseek-v3.2", false, 0, 0},
		{"deepseek-v3.2:cloud", true, 163_840, 65_536},
		{"qwen3.5", false, 0, 0},
		{"qwen3.5:cloud", true, 262_144, 32_768},
		{"qwen3-coder:480b", false, 0, 0},
		{"qwen3-coder:480b:cloud", true, 262_144, 65_536},
		{"qwen3-coder-next:cloud", true, 262_144, 32_768},
		{"llama3.2", false, 0, 0},
		{"unknown-model:cloud", false, 0, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l, ok := lookupCloudModelLimit(tt.name)
			if ok != tt.wantOK {
				t.Errorf("lookupCloudModelLimit(%q) ok = %v, want %v", tt.name, ok, tt.wantOK)
			}
			if ok {
				if l.Context != tt.wantContext {
					t.Errorf("context = %d, want %d", l.Context, tt.wantContext)
				}
				if l.Output != tt.wantOutput {
					t.Errorf("output = %d, want %d", l.Output, tt.wantOutput)
				}
			}
		})
	}
}

func TestFindOpenCode(t *testing.T) {
	t.Run("fallback to ~/.opencode/bin", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		// Ensure opencode is not on PATH
		t.Setenv("PATH", tmpDir)

		// Without the fallback binary, findOpenCode should fail
		if _, ok := findOpenCode(); ok {
			t.Fatal("findOpenCode should fail when binary is not on PATH or in fallback location")
		}

		// Create a fake binary at the curl install fallback location
		binDir := filepath.Join(tmpDir, ".opencode", "bin")
		os.MkdirAll(binDir, 0o755)
		name := "opencode"
		if runtime.GOOS == "windows" {
			name = "opencode.exe"
		}
		fakeBin := filepath.Join(binDir, name)
		os.WriteFile(fakeBin, []byte("#!/bin/sh\n"), 0o755)

		// Now findOpenCode should succeed via fallback
		path, ok := findOpenCode()
		if !ok {
			t.Fatal("findOpenCode should succeed with fallback binary")
		}
		if path != fakeBin {
			t.Errorf("findOpenCode = %q, want %q", path, fakeBin)
		}
	})
}

// Verify that the BackfillsCloudModelLimitOnExistingEntry test from the old
// file-based approach is covered by the new inline config approach.
func TestOpenCodeEdit_CloudModelLimitStructure(t *testing.T) {
	o := &OpenCode{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	expected := cloudModelLimits["glm-4.7"]

	if err := o.Edit([]string{"glm-4.7:cloud"}); err != nil {
		t.Fatal(err)
	}

	var cfg map[string]any
	json.Unmarshal([]byte(o.configContent), &cfg)
	provider, _ := cfg["provider"].(map[string]any)
	ollama, _ := provider["ollama"].(map[string]any)
	models, _ := ollama["models"].(map[string]any)
	entry, _ := models["glm-4.7:cloud"].(map[string]any)

	limit, ok := entry["limit"].(map[string]any)
	if !ok {
		t.Fatal("cloud model limit was not set")
	}
	if limit["context"] != float64(expected.Context) {
		t.Errorf("context = %v, want %d", limit["context"], expected.Context)
	}
	if limit["output"] != float64(expected.Output) {
		t.Errorf("output = %v, want %d", limit["output"], expected.Output)
	}
}

func TestOpenCodeEdit_SpecialCharsInModelName(t *testing.T) {
	o := &OpenCode{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	specialModel := `model-with-"quotes"`

	err := o.Edit([]string{specialModel})
	if err != nil {
		t.Fatalf("Edit with special chars failed: %v", err)
	}

	var cfg map[string]any
	if err := json.Unmarshal([]byte(o.configContent), &cfg); err != nil {
		t.Fatalf("resulting config is invalid JSON: %v", err)
	}

	provider, _ := cfg["provider"].(map[string]any)
	ollama, _ := provider["ollama"].(map[string]any)
	models, _ := ollama["models"].(map[string]any)
	if models[specialModel] == nil {
		t.Errorf("model with special chars not found in config")
	}
}

func TestReadModelJSONModels(t *testing.T) {
	t.Run("reads ollama models from model.json", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		stateDir := filepath.Join(tmpDir, ".local", "state", "opencode")
		os.MkdirAll(stateDir, 0o755)
		state := map[string]any{
			"recent": []any{
				map[string]any{"providerID": "ollama", "modelID": "llama3.2"},
				map[string]any{"providerID": "ollama", "modelID": "qwen3:32b"},
			},
		}
		data, _ := json.MarshalIndent(state, "", "  ")
		os.WriteFile(filepath.Join(stateDir, "model.json"), data, 0o644)

		models := readModelJSONModels()
		if len(models) != 2 {
			t.Fatalf("got %d models, want 2", len(models))
		}
		if models[0] != "llama3.2" || models[1] != "qwen3:32b" {
			t.Errorf("got %v, want [llama3.2 qwen3:32b]", models)
		}
	})

	t.Run("skips non-ollama providers", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		stateDir := filepath.Join(tmpDir, ".local", "state", "opencode")
		os.MkdirAll(stateDir, 0o755)
		state := map[string]any{
			"recent": []any{
				map[string]any{"providerID": "openai", "modelID": "gpt-4"},
				map[string]any{"providerID": "ollama", "modelID": "llama3.2"},
			},
		}
		data, _ := json.MarshalIndent(state, "", "  ")
		os.WriteFile(filepath.Join(stateDir, "model.json"), data, 0o644)

		models := readModelJSONModels()
		if len(models) != 1 || models[0] != "llama3.2" {
			t.Errorf("got %v, want [llama3.2]", models)
		}
	})

	t.Run("returns nil when file does not exist", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		if models := readModelJSONModels(); models != nil {
			t.Errorf("got %v, want nil", models)
		}
	})

	t.Run("returns nil for corrupt JSON", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		stateDir := filepath.Join(tmpDir, ".local", "state", "opencode")
		os.MkdirAll(stateDir, 0o755)
		os.WriteFile(filepath.Join(stateDir, "model.json"), []byte(`{corrupt`), 0o644)

		if models := readModelJSONModels(); models != nil {
			t.Errorf("got %v, want nil", models)
		}
	})
}

func TestOpenCodeResolveContent(t *testing.T) {
	t.Run("returns Edit's content when set", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		o := &OpenCode{}
		if err := o.Edit([]string{"gemma4"}); err != nil {
			t.Fatal(err)
		}
		editContent := o.configContent

		// Write a different model.json — should be ignored
		stateDir := filepath.Join(tmpDir, ".local", "state", "opencode")
		state := map[string]any{
			"recent": []any{
				map[string]any{"providerID": "ollama", "modelID": "different-model"},
			},
		}
		data, _ := json.MarshalIndent(state, "", "  ")
		os.WriteFile(filepath.Join(stateDir, "model.json"), data, 0o644)

		got := o.resolveContent("gemma4")
		if got != editContent {
			t.Errorf("resolveContent returned different content than Edit set\ngot:  %s\nwant: %s", got, editContent)
		}
	})

	t.Run("falls back to model.json when Edit was not called", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		stateDir := filepath.Join(tmpDir, ".local", "state", "opencode")
		os.MkdirAll(stateDir, 0o755)
		state := map[string]any{
			"recent": []any{
				map[string]any{"providerID": "ollama", "modelID": "llama3.2"},
				map[string]any{"providerID": "ollama", "modelID": "qwen3:32b"},
			},
		}
		data, _ := json.MarshalIndent(state, "", "  ")
		os.WriteFile(filepath.Join(stateDir, "model.json"), data, 0o644)

		o := &OpenCode{}
		content := o.resolveContent("llama3.2")
		if content == "" {
			t.Fatal("resolveContent returned empty")
		}

		var cfg map[string]any
		json.Unmarshal([]byte(content), &cfg)
		if cfg["model"] != "ollama/llama3.2" {
			t.Errorf("primary = %v, want ollama/llama3.2", cfg["model"])
		}
		provider, _ := cfg["provider"].(map[string]any)
		ollama, _ := provider["ollama"].(map[string]any)
		cfgModels, _ := ollama["models"].(map[string]any)
		if cfgModels["llama3.2"] == nil || cfgModels["qwen3:32b"] == nil {
			t.Errorf("expected both models in config, got %v", cfgModels)
		}
	})

	t.Run("uses requested model as primary even when not first in model.json", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		stateDir := filepath.Join(tmpDir, ".local", "state", "opencode")
		os.MkdirAll(stateDir, 0o755)
		state := map[string]any{
			"recent": []any{
				map[string]any{"providerID": "ollama", "modelID": "llama3.2"},
				map[string]any{"providerID": "ollama", "modelID": "qwen3:32b"},
			},
		}
		data, _ := json.MarshalIndent(state, "", "  ")
		os.WriteFile(filepath.Join(stateDir, "model.json"), data, 0o644)

		o := &OpenCode{}
		content := o.resolveContent("qwen3:32b")

		var cfg map[string]any
		json.Unmarshal([]byte(content), &cfg)
		if cfg["model"] != "ollama/qwen3:32b" {
			t.Errorf("primary = %v, want ollama/qwen3:32b", cfg["model"])
		}
	})

	t.Run("injects requested model when missing from model.json", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		stateDir := filepath.Join(tmpDir, ".local", "state", "opencode")
		os.MkdirAll(stateDir, 0o755)
		state := map[string]any{
			"recent": []any{
				map[string]any{"providerID": "ollama", "modelID": "llama3.2"},
			},
		}
		data, _ := json.MarshalIndent(state, "", "  ")
		os.WriteFile(filepath.Join(stateDir, "model.json"), data, 0o644)

		o := &OpenCode{}
		content := o.resolveContent("gemma4")

		var cfg map[string]any
		json.Unmarshal([]byte(content), &cfg)
		provider, _ := cfg["provider"].(map[string]any)
		ollama, _ := provider["ollama"].(map[string]any)
		cfgModels, _ := ollama["models"].(map[string]any)
		if cfgModels["gemma4"] == nil {
			t.Error("requested model gemma4 not injected into config")
		}
		if cfg["model"] != "ollama/gemma4" {
			t.Errorf("primary = %v, want ollama/gemma4", cfg["model"])
		}
	})

	t.Run("returns empty when no model.json and no model param", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		o := &OpenCode{}
		if got := o.resolveContent(""); got != "" {
			t.Errorf("resolveContent(\"\") = %q, want empty", got)
		}
	})

	t.Run("does not mutate configContent on fallback", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		stateDir := filepath.Join(tmpDir, ".local", "state", "opencode")
		os.MkdirAll(stateDir, 0o755)
		state := map[string]any{
			"recent": []any{
				map[string]any{"providerID": "ollama", "modelID": "llama3.2"},
			},
		}
		data, _ := json.MarshalIndent(state, "", "  ")
		os.WriteFile(filepath.Join(stateDir, "model.json"), data, 0o644)

		o := &OpenCode{}
		_ = o.resolveContent("llama3.2")
		if o.configContent != "" {
			t.Errorf("resolveContent should not mutate configContent, got %q", o.configContent)
		}
	})
}

func TestBuildInlineConfig(t *testing.T) {
	t.Run("returns error for empty primary", func(t *testing.T) {
		if _, err := buildInlineConfig("", []string{"llama3.2"}); err == nil {
			t.Error("expected error for empty primary")
		}
	})

	t.Run("returns error for empty models", func(t *testing.T) {
		if _, err := buildInlineConfig("llama3.2", nil); err == nil {
			t.Error("expected error for empty models")
		}
	})

	t.Run("primary differs from first model in list", func(t *testing.T) {
		content, err := buildInlineConfig("qwen3:32b", []string{"llama3.2", "qwen3:32b"})
		if err != nil {
			t.Fatal(err)
		}
		var cfg map[string]any
		json.Unmarshal([]byte(content), &cfg)
		if cfg["model"] != "ollama/qwen3:32b" {
			t.Errorf("primary = %v, want ollama/qwen3:32b", cfg["model"])
		}
	})
}

func TestOpenCodeEdit_PreservesRecentEntries(t *testing.T) {
	t.Run("prepends new models to existing recent", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		stateDir := filepath.Join(tmpDir, ".local", "state", "opencode")
		os.MkdirAll(stateDir, 0o755)
		initial := map[string]any{
			"recent": []any{
				map[string]any{"providerID": "ollama", "modelID": "old-A"},
				map[string]any{"providerID": "ollama", "modelID": "old-B"},
			},
		}
		data, _ := json.MarshalIndent(initial, "", "  ")
		os.WriteFile(filepath.Join(stateDir, "model.json"), data, 0o644)

		o := &OpenCode{}
		if err := o.Edit([]string{"new-X"}); err != nil {
			t.Fatal(err)
		}

		stored, _ := os.ReadFile(filepath.Join(stateDir, "model.json"))
		var state map[string]any
		json.Unmarshal(stored, &state)
		recent, _ := state["recent"].([]any)

		if len(recent) != 3 {
			t.Fatalf("expected 3 entries, got %d", len(recent))
		}
		first, _ := recent[0].(map[string]any)
		if first["modelID"] != "new-X" {
			t.Errorf("first entry = %v, want new-X", first["modelID"])
		}
	})

	t.Run("prepends multiple new models in order", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		stateDir := filepath.Join(tmpDir, ".local", "state", "opencode")
		os.MkdirAll(stateDir, 0o755)
		initial := map[string]any{
			"recent": []any{
				map[string]any{"providerID": "ollama", "modelID": "old-A"},
				map[string]any{"providerID": "ollama", "modelID": "old-B"},
			},
		}
		data, _ := json.MarshalIndent(initial, "", "  ")
		os.WriteFile(filepath.Join(stateDir, "model.json"), data, 0o644)

		o := &OpenCode{}
		if err := o.Edit([]string{"X", "Y", "Z"}); err != nil {
			t.Fatal(err)
		}

		stored, _ := os.ReadFile(filepath.Join(stateDir, "model.json"))
		var state map[string]any
		json.Unmarshal(stored, &state)
		recent, _ := state["recent"].([]any)

		want := []string{"X", "Y", "Z", "old-A", "old-B"}
		if len(recent) != len(want) {
			t.Fatalf("expected %d entries, got %d", len(want), len(recent))
		}
		for i, w := range want {
			e, _ := recent[i].(map[string]any)
			if e["modelID"] != w {
				t.Errorf("recent[%d] = %v, want %v", i, e["modelID"], w)
			}
		}
	})

	t.Run("preserves non-ollama entries", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		stateDir := filepath.Join(tmpDir, ".local", "state", "opencode")
		os.MkdirAll(stateDir, 0o755)
		initial := map[string]any{
			"recent": []any{
				map[string]any{"providerID": "openai", "modelID": "gpt-4"},
				map[string]any{"providerID": "ollama", "modelID": "llama3.2"},
			},
		}
		data, _ := json.MarshalIndent(initial, "", "  ")
		os.WriteFile(filepath.Join(stateDir, "model.json"), data, 0o644)

		o := &OpenCode{}
		if err := o.Edit([]string{"qwen3:32b"}); err != nil {
			t.Fatal(err)
		}

		stored, _ := os.ReadFile(filepath.Join(stateDir, "model.json"))
		var state map[string]any
		json.Unmarshal(stored, &state)
		recent, _ := state["recent"].([]any)

		// Should have: qwen3:32b (new), gpt-4 (preserved openai), llama3.2 (preserved ollama)
		var foundOpenAI bool
		for _, entry := range recent {
			e, _ := entry.(map[string]any)
			if e["providerID"] == "openai" && e["modelID"] == "gpt-4" {
				foundOpenAI = true
			}
		}
		if !foundOpenAI {
			t.Errorf("non-ollama gpt-4 entry was not preserved, got %v", recent)
		}
	})

	t.Run("deduplicates ollama models being re-added", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		stateDir := filepath.Join(tmpDir, ".local", "state", "opencode")
		os.MkdirAll(stateDir, 0o755)
		initial := map[string]any{
			"recent": []any{
				map[string]any{"providerID": "ollama", "modelID": "llama3.2"},
			},
		}
		data, _ := json.MarshalIndent(initial, "", "  ")
		os.WriteFile(filepath.Join(stateDir, "model.json"), data, 0o644)

		o := &OpenCode{}
		if err := o.Edit([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}

		stored, _ := os.ReadFile(filepath.Join(stateDir, "model.json"))
		var state map[string]any
		json.Unmarshal(stored, &state)
		recent, _ := state["recent"].([]any)

		count := 0
		for _, entry := range recent {
			e, _ := entry.(map[string]any)
			if e["modelID"] == "llama3.2" {
				count++
			}
		}
		if count != 1 {
			t.Errorf("expected 1 llama3.2 entry, got %d", count)
		}
	})

	t.Run("caps recent list at 10", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		stateDir := filepath.Join(tmpDir, ".local", "state", "opencode")
		os.MkdirAll(stateDir, 0o755)

		// Pre-populate with 9 distinct ollama models
		recentEntries := make([]any, 0, 9)
		for i := range 9 {
			recentEntries = append(recentEntries, map[string]any{
				"providerID": "ollama",
				"modelID":    fmt.Sprintf("old-%d", i),
			})
		}
		initial := map[string]any{"recent": recentEntries}
		data, _ := json.MarshalIndent(initial, "", "  ")
		os.WriteFile(filepath.Join(stateDir, "model.json"), data, 0o644)

		// Add 5 new models — should cap at 10 total
		o := &OpenCode{}
		if err := o.Edit([]string{"new-0", "new-1", "new-2", "new-3", "new-4"}); err != nil {
			t.Fatal(err)
		}

		stored, _ := os.ReadFile(filepath.Join(stateDir, "model.json"))
		var state map[string]any
		json.Unmarshal(stored, &state)
		recent, _ := state["recent"].([]any)

		if len(recent) != 10 {
			t.Errorf("expected 10 entries (capped), got %d", len(recent))
		}
	})
}

func TestOpenCodeEdit_BaseURL(t *testing.T) {
	o := &OpenCode{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	// Default OLLAMA_HOST
	o.Edit([]string{"llama3.2"})

	var cfg map[string]any
	json.Unmarshal([]byte(o.configContent), &cfg)
	provider, _ := cfg["provider"].(map[string]any)
	ollama, _ := provider["ollama"].(map[string]any)
	options, _ := ollama["options"].(map[string]any)

	baseURL, _ := options["baseURL"].(string)
	if baseURL == "" {
		t.Error("baseURL should be set")
	}
}
