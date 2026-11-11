package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPrimaryModelSelectionPreservesUnownedFields(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	path, err := configPath()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	fixture := `{"theme":"dark","future":{"keep":true},"integrations":{"chatgpt":{"models":["second","first"],"primary":"first","onboarded":true,"aliases":{"keep":"original"},"catalog":{"revision":7}},"codex":{"models":["terminal"]},"claude-desktop":{"models":["claude-choice"]}}}`
	if err := os.WriteFile(path, []byte(fixture), 0o600); err != nil {
		t.Fatal(err)
	}
	before, err := LoadIntegration("chatgpt")
	if err != nil {
		t.Fatal(err)
	}
	applied := &IntegrationConfig{Models: before.Models, Primary: "second"}
	if err := SwapPrimaryModelSelection("chatgpt", before, applied); err != nil {
		t.Fatal(err)
	}
	// An unrelated edit made during the attempt must survive its rollback.
	if err := SaveAliases("chatgpt", map[string]string{"keep": "later"}); err != nil {
		t.Fatal(err)
	}
	if err := SwapPrimaryModelSelection("chatgpt", applied, before); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	var want map[string]any
	if err := json.Unmarshal([]byte(strings.Replace(fixture, `"keep":"original"`, `"keep":"later"`, 1)), &want); err != nil {
		t.Fatal(err)
	}
	gotJSON, _ := json.Marshal(got)
	wantJSON, _ := json.Marshal(want)
	if string(gotJSON) != string(wantJSON) {
		t.Fatalf("rollback changed unowned fields: got=%s want=%s", gotJSON, wantJSON)
	}
}

func TestPrimaryModelSelectionRollbackProtectsLaterPrimary(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	before := &IntegrationConfig{Models: []string{"first", "second"}, Primary: "first"}
	applied := &IntegrationConfig{Models: before.Models, Primary: "second"}
	later := &IntegrationConfig{Models: before.Models, Primary: "later-primary"}
	for _, step := range []struct{ before, after *IntegrationConfig }{{nil, before}, {before, applied}, {applied, later}} {
		if err := SwapPrimaryModelSelection("chatgpt", step.before, step.after); err != nil {
			t.Fatal(err)
		}
	}
	if err := SwapPrimaryModelSelection("chatgpt", applied, before); err == nil {
		t.Fatal("rollback overwrote a later primary change")
	}
	got, err := LoadIntegration("chatgpt")
	if err != nil || got.Primary != later.Primary {
		t.Fatalf("later primary=%+v error=%v", got, err)
	}
}
