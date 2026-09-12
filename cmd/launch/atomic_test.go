package launch

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func atomicTestConfig(t *testing.T) map[string]any {
	t.Helper()
	path, err := atomicConfigPath()
	if err != nil {
		t.Fatalf("config path: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read config: %v", err)
	}
	config := make(map[string]any)
	if err := json.Unmarshal(data, &config); err != nil {
		t.Fatalf("parse config: %v", err)
	}
	return config
}

func atomicTestProviders(t *testing.T, config map[string]any) []map[string]any {
	t.Helper()
	llm, _ := config["llm"].(map[string]any)
	if llm == nil {
		t.Fatalf("config has no llm block: %v", config)
	}
	raw, _ := llm["providers"].([]any)
	out := make([]map[string]any, 0, len(raw))
	for _, item := range raw {
		entry, _ := item.(map[string]any)
		if entry == nil {
			t.Fatalf("provider entry is not an object: %v", item)
		}
		out = append(out, entry)
	}
	return out
}

func atomicTestProvider(t *testing.T, config map[string]any, id string) map[string]any {
	t.Helper()
	for _, entry := range atomicTestProviders(t, config) {
		if entry["id"] == id {
			return entry
		}
	}
	t.Fatalf("provider %q not found", id)
	return nil
}

func TestAtomicEditCreatesConfig(t *testing.T) {
	t.Setenv("ATOMIC_AGENT_STATE_DIR", t.TempDir())

	a := &Atomic{}
	if err := a.Edit([]LaunchModel{{Name: "qwen3.6"}}); err != nil {
		t.Fatalf("edit: %v", err)
	}

	config := atomicTestConfig(t)
	llm := config["llm"].(map[string]any)
	if llm["activeTextProvider"] != atomicLaunchProviderID {
		t.Fatalf("activeTextProvider = %v", llm["activeTextProvider"])
	}
	if llm["activeEmbeddingProvider"] != "local-llama" {
		t.Fatalf("activeEmbeddingProvider = %v", llm["activeEmbeddingProvider"])
	}
	if llm["toolTransport"] != "auto" {
		t.Fatalf("toolTransport = %v", llm["toolTransport"])
	}

	// The referenced embedding provider must exist or the agent's
	// config validation rejects the whole file.
	atomicTestProvider(t, config, "local-llama")

	entry := atomicTestProvider(t, config, atomicLaunchProviderID)
	if entry["kind"] != "openai-compatible" {
		t.Fatalf("kind = %v", entry["kind"])
	}
	if entry["defaultChatModel"] != "qwen3.6" {
		t.Fatalf("defaultChatModel = %v", entry["defaultChatModel"])
	}
	if entry["apiKeyEnvVar"] != "OLLAMA_API_KEY" {
		t.Fatalf("apiKeyEnvVar = %v", entry["apiKeyEnvVar"])
	}
	baseURL, _ := entry["baseUrl"].(string)
	if baseURL != atomicOllamaBaseURL() {
		t.Fatalf("baseUrl = %q, want %q", baseURL, atomicOllamaBaseURL())
	}
	// Atomic Agent stores compat base URLs without the /v1 suffix; a
	// stored suffix would produce /v1/v1/... requests, which 404.
	if strings.HasSuffix(baseURL, "/v1") {
		t.Fatalf("baseUrl %q must not end in /v1", baseURL)
	}
}

func TestAtomicEditPreservesExistingConfig(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("ATOMIC_AGENT_STATE_DIR", dir)

	existing := map[string]any{
		"version":   float64(41),
		"telemetry": map[string]any{"enabled": false},
		"llm": map[string]any{
			"activeTextProvider":      "groq",
			"activeEmbeddingProvider": "groq",
			"toolTransport":           "native_tools",
			"providers": []any{
				map[string]any{
					"id":               "groq",
					"kind":             "openai-compatible",
					"baseUrl":          "https://api.groq.com/openai",
					"defaultChatModel": "llama-3.3-70b-versatile",
				},
			},
		},
	}
	data, err := json.Marshal(existing)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}

	a := &Atomic{}
	if err := a.Edit([]LaunchModel{{Name: "gemma4"}}); err != nil {
		t.Fatalf("edit: %v", err)
	}

	config := atomicTestConfig(t)
	if config["version"] != float64(41) {
		t.Fatalf("version was not preserved: %v", config["version"])
	}
	telemetry, _ := config["telemetry"].(map[string]any)
	if telemetry == nil || telemetry["enabled"] != false {
		t.Fatalf("telemetry block was not preserved: %v", config["telemetry"])
	}

	llm := config["llm"].(map[string]any)
	if llm["activeTextProvider"] != atomicLaunchProviderID {
		t.Fatalf("activeTextProvider = %v", llm["activeTextProvider"])
	}
	// Values the launcher does not own stay untouched.
	if llm["activeEmbeddingProvider"] != "groq" {
		t.Fatalf("activeEmbeddingProvider = %v", llm["activeEmbeddingProvider"])
	}
	if llm["toolTransport"] != "native_tools" {
		t.Fatalf("toolTransport = %v", llm["toolTransport"])
	}

	groq := atomicTestProvider(t, config, "groq")
	if groq["defaultChatModel"] != "llama-3.3-70b-versatile" {
		t.Fatalf("groq entry was modified: %v", groq)
	}
	entry := atomicTestProvider(t, config, atomicLaunchProviderID)
	if entry["defaultChatModel"] != "gemma4" {
		t.Fatalf("defaultChatModel = %v", entry["defaultChatModel"])
	}
}

func TestAtomicEditUpdatesExistingEntryInPlace(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("ATOMIC_AGENT_STATE_DIR", dir)

	a := &Atomic{}
	if err := a.Edit([]LaunchModel{{Name: "qwen3.6"}}); err != nil {
		t.Fatalf("first edit: %v", err)
	}

	// Simulate the agent adding its own key to the entry between runs.
	config := atomicTestConfig(t)
	entry := atomicTestProvider(t, config, atomicLaunchProviderID)
	entry["requestTimeoutMs"] = float64(120000)
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}

	if err := a.Edit([]LaunchModel{{Name: "gemma4:e4b"}}); err != nil {
		t.Fatalf("second edit: %v", err)
	}

	config = atomicTestConfig(t)
	providers := atomicTestProviders(t, config)
	count := 0
	for _, p := range providers {
		if p["id"] == atomicLaunchProviderID {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("expected one launcher entry, found %d", count)
	}
	entry = atomicTestProvider(t, config, atomicLaunchProviderID)
	if entry["defaultChatModel"] != "gemma4:e4b" {
		t.Fatalf("defaultChatModel = %v", entry["defaultChatModel"])
	}
	if entry["requestTimeoutMs"] != float64(120000) {
		t.Fatalf("agent-owned key was dropped: %v", entry)
	}
}

func TestAtomicEditWithoutModelsIsNoOp(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("ATOMIC_AGENT_STATE_DIR", dir)

	a := &Atomic{}
	if err := a.Edit(nil); err != nil {
		t.Fatalf("edit: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "config.json")); !os.IsNotExist(err) {
		t.Fatalf("config should not have been created, stat err = %v", err)
	}
}

func TestAtomicModels(t *testing.T) {
	t.Setenv("ATOMIC_AGENT_STATE_DIR", t.TempDir())

	a := &Atomic{}
	if models := a.Models(); models != nil {
		t.Fatalf("expected nil before configuration, got %v", models)
	}
	if err := a.Edit([]LaunchModel{{Name: "qwen3.6"}}); err != nil {
		t.Fatalf("edit: %v", err)
	}
	models := a.Models()
	if len(models) != 1 || models[0] != "qwen3.6" {
		t.Fatalf("models = %v", models)
	}
}

func TestAtomicModelsIgnoresForeignActiveProvider(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("ATOMIC_AGENT_STATE_DIR", dir)

	a := &Atomic{}
	if err := a.Edit([]LaunchModel{{Name: "qwen3.6"}}); err != nil {
		t.Fatalf("edit: %v", err)
	}
	config := atomicTestConfig(t)
	config["llm"].(map[string]any)["activeTextProvider"] = "groq"
	data, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}

	if models := a.Models(); models != nil {
		t.Fatalf("expected nil when another provider is active, got %v", models)
	}
}

func TestAtomicFindBinaryHonorsInstallDirOverride(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix executable bit")
	}
	dir := t.TempDir()
	bin := filepath.Join(dir, "atomic-agent")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ATOMIC_AGENT_INSTALL_DIR", dir)
	t.Setenv("PATH", t.TempDir())

	found, err := findAtomicAgent()
	if err != nil {
		t.Fatalf("findAtomicAgent: %v", err)
	}
	if found != bin {
		t.Fatalf("found %q, want %q", found, bin)
	}
}
