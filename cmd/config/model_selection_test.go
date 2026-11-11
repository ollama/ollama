package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/cmd/internal/fileutil"
)

func TestPrimaryModelSelectionPreservesUnknownFieldsAndConditionalRollback(t *testing.T) {
	setTestHome(t, t.TempDir())
	path, _ := configPath()
	os.MkdirAll(filepath.Dir(path), 0700)
	original := `{"future":{"enabled":true},"integrations":{"other":{"models":["keep"],"secret":"synthetic"},"chatgpt":{"models":["old"],"primary":"old","aliases":{"keep":"original"},"automode":false,"onboarded":true,"future":"keep"}}}`
	if err := os.WriteFile(path, []byte(original), 0600); err != nil {
		t.Fatal(err)
	}
	before := &IntegrationConfig{Models: []string{"old"}, Primary: "old"}
	applied := &IntegrationConfig{Models: []string{"new"}, Primary: "new"}
	if err := SwapPrimaryModelSelection("chatgpt", before, applied); err != nil {
		t.Fatal(err)
	}
	if err := SaveIntegrationAutoMode("chatgpt", true); err != nil {
		t.Fatal(err)
	}
	var document map[string]any
	data, _ := os.ReadFile(path)
	if err := json.Unmarshal(data, &document); err != nil {
		t.Fatal(err)
	}
	integrations := document["integrations"].(map[string]any)
	entry := integrations["chatgpt"].(map[string]any)
	if document["future"] == nil || entry["automode"] != true || entry["onboarded"] != true || entry["future"] != "keep" || entry["aliases"].(map[string]any)["keep"] != "original" || integrations["other"].(map[string]any)["secret"] != "synthetic" {
		t.Fatalf("unrelated data changed: %s", data)
	}
	if err := SwapPrimaryModelSelection("chatgpt", applied, before); err != nil {
		t.Fatal(err)
	}
	restored, err := LoadIntegration("chatgpt")
	if err != nil || restored.AutoMode == nil || !*restored.AutoMode {
		t.Fatalf("selection rollback changed the later auto-mode preference: %+v, %v", restored, err)
	}
	if err := SwapPrimaryModelSelection("chatgpt", before, &IntegrationConfig{Models: []string{"later-user-choice"}}); err != nil {
		t.Fatal(err)
	}
	if err := SwapPrimaryModelSelection("chatgpt", applied, before); err == nil {
		t.Fatal("rollback replaced a later user change")
	}
	current, err := LoadIntegration("chatgpt")
	if err != nil || len(current.Models) != 1 || current.Models[0] != "later-user-choice" {
		t.Fatalf("current = %+v, %v", current, err)
	}
}

func TestPrimaryModelSelectionConcurrentLauncherUpdatesPreserveAllFields(t *testing.T) {
	setTestHome(t, t.TempDir())
	path, _ := configPath()
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		t.Fatal(err)
	}
	original := `{"future":"keep","integrations":{"chatgpt":{"models":["old"],"aliases":{"keep":"original"},"automode":true,"future":"keep"},"codex":{"models":["terminal"],"future":"keep"}}}`
	if err := os.WriteFile(path, []byte(original), 0600); err != nil {
		t.Fatal(err)
	}
	for i := range 8 {
		model := fmt.Sprintf("local-%d", i)
		enabled := i%2 != 0
		before, err := LoadIntegration("chatgpt")
		if err != nil {
			t.Fatal(err)
		}
		start := make(chan struct{})
		results := make(chan error, 8)
		for _, update := range []func() error{
			func() error {
				return SwapPrimaryModelSelection("chatgpt", before, &IntegrationConfig{Models: []string{model}, Primary: model})
			},
			func() error { return SaveIntegrationAutoMode("chatgpt", enabled) },
			func() error { return SetLastSelection("chatgpt") },
			func() error { return SetLastModel("chat-" + model) },
			func() error { return SaveIntegration("codex", []string{"terminal-" + model}) },
			func() error { return SaveAliases("codex", map[string]string{"helper": "alias-" + model}) },
			func() error { return MarkIntegrationOnboarded("codex") },
			func() error { return completeWelcomeInConfig() },
		} {
			go func(update func() error) { <-start; results <- update() }(update)
		}
		close(start)
		var failures []error
		for range 8 {
			if err := <-results; err != nil {
				failures = append(failures, err)
			}
		}
		if len(failures) > 0 {
			t.Fatal(failures)
		}
		current, err := LoadIntegration("chatgpt")
		if LastSelection() != "chatgpt" || err != nil || len(current.Models) != 1 || current.Models[0] != model || current.Primary != model || current.Aliases["keep"] != "original" || current.AutoMode == nil || *current.AutoMode != enabled {
			t.Fatalf("concurrent update was lost: %+v, %v", current, err)
		}

		terminal, err := LoadIntegration("codex")
		if err != nil || len(terminal.Models) != 1 || terminal.Models[0] != "terminal-"+model || terminal.Aliases["helper"] != "alias-"+model || !terminal.Onboarded || LastModel() != "chat-"+model {
			t.Fatalf("other launcher update lost: %+v last_model=%s error=%v", terminal, LastModel(), err)
		}
	}
	data, _ := os.ReadFile(path)
	var document map[string]any
	if err := json.Unmarshal(data, &document); err != nil {
		t.Fatal(err)
	}
	integrations := document["integrations"].(map[string]any)
	if document["future"] != "keep" || integrations["chatgpt"].(map[string]any)["future"] != "keep" || integrations["codex"].(map[string]any)["future"] != "keep" {
		t.Fatalf("concurrent updates lost unrelated fields: %s", data)
	}
}

func TestPrimaryModelSelectionRemovesNewEntry(t *testing.T) {
	setTestHome(t, t.TempDir())
	applied := &IntegrationConfig{Models: []string{"local"}}
	if err := SwapPrimaryModelSelection("chatgpt", nil, applied); err != nil {
		t.Fatal(err)
	}
	if err := SwapPrimaryModelSelection("chatgpt", applied, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadIntegration("chatgpt"); !os.IsNotExist(err) {
		t.Fatalf("new entry survived rollback: %v", err)
	}
}

func TestLauncherPreferenceWriteCooperatesWithSelectionLock(t *testing.T) {
	setTestHome(t, t.TempDir())
	if err := SwapPrimaryModelSelection("chatgpt", nil, &IntegrationConfig{Models: []string{"local"}, Primary: "local"}); err != nil {
		t.Fatal(err)
	}
	path, _ := configPath()
	unlock, err := fileutil.Lock(path + ".lock")
	if err != nil {
		t.Fatal(err)
	}
	released := false
	defer func() {
		if !released {
			unlock()
		}
	}()
	result := make(chan error, 1)
	go func() { result <- SetLastModel("chat-model") }()
	select {
	case err := <-result:
		t.Fatalf("preference write bypassed the held model-selection lock: %v", err)
	case <-time.After(50 * time.Millisecond):
	}
	unlock()
	released = true
	if err := <-result; err != nil {
		t.Fatal(err)
	}
	if LastModel() != "chat-model" || IntegrationModel("chatgpt") != "local" {
		t.Fatal("serialized writes lost settings")
	}
}

func TestLegacyMigrationPreservesConfigWrittenWhileWaiting(t *testing.T) {
	setTestHome(t, t.TempDir())
	path, _ := configPath()
	legacy, _ := legacyConfigPath()
	if err := os.MkdirAll(filepath.Dir(legacy), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(legacy, []byte(`{"integrations":{"chatgpt":{"models":["old"]}}}`), 0600); err != nil {
		t.Fatal(err)
	}
	unlock, err := fileutil.Lock(path + ".lock")
	if err != nil {
		t.Fatal(err)
	}
	released := false
	defer func() {
		if !released {
			unlock()
		}
	}()
	type loaded struct {
		cfg *config
		err error
	}
	result := make(chan loaded, 1)
	go func() { cfg, err := load(); result <- loaded{cfg, err} }()
	select {
	case got := <-result:
		t.Fatalf("legacy migration bypassed the held config lock: %+v", got)
	case <-time.After(50 * time.Millisecond):
	}
	current := []byte(`{"future":"keep","integrations":{"chatgpt":{"models":["new"]}}}`)
	if err := fileutil.WriteWithBackup(path, current); err != nil {
		t.Fatal(err)
	}
	unlock()
	released = true
	got := <-result
	if got.err != nil || got.cfg.Integrations["chatgpt"] == nil || len(got.cfg.Integrations["chatgpt"].Models) != 1 || got.cfg.Integrations["chatgpt"].Models[0] != "new" {
		t.Fatalf("migration replaced or failed to reread newer config: %+v/%v", got.cfg, got.err)
	}
	data, err := os.ReadFile(path)
	if err != nil || string(data) != string(current) {
		t.Fatalf("migration replaced newer document: %s/%v", data, err)
	}
}
