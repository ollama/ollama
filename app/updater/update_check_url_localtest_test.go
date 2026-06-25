//go:build (windows || darwin) && updater_localtest

package updater

import "testing"

func TestUpdateCheckURLBaseUsesLocalTestEnv(t *testing.T) {
	t.Setenv("OLLAMA_TEST_UPDATE_URL", "http://127.0.0.1:8765/api/update")

	old := UpdateCheckURLBase
	UpdateCheckURLBase = defaultUpdateCheckURLBase
	t.Cleanup(func() {
		UpdateCheckURLBase = old
	})

	if got := updateCheckURLBase(); got != "http://127.0.0.1:8765/api/update" {
		t.Fatalf("updateCheckURLBase() = %q", got)
	}
}

func TestUpdateCheckURLBaseUsesExplicitTestOverrideBeforeLocalTestEnv(t *testing.T) {
	t.Setenv("OLLAMA_TEST_UPDATE_URL", "http://127.0.0.1:8765/api/update")

	old := UpdateCheckURLBase
	UpdateCheckURLBase = "http://127.0.0.1:8765/update.json"
	t.Cleanup(func() {
		UpdateCheckURLBase = old
	})

	if got := updateCheckURLBase(); got != "http://127.0.0.1:8765/update.json" {
		t.Fatalf("updateCheckURLBase() = %q", got)
	}
}
