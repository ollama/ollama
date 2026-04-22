package launch

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type appLaunchIntegration struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Command     string `json:"command"`
	Description string `json:"description"`
}

func loadAppLaunchManifest(t *testing.T) []appLaunchIntegration {
	t.Helper()

	path := filepath.Join("..", "..", "app", "ui", "app", "src", "data", "launch-integrations.ts")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read app launch manifest: %v", err)
	}

	jsonData := extractJSONArray(t, string(data))

	var manifest []appLaunchIntegration
	if err := json.Unmarshal([]byte(jsonData), &manifest); err != nil {
		t.Fatalf("parse app launch manifest: %v", err)
	}

	return manifest
}

func extractJSONArray(t *testing.T, src string) string {
	t.Helper()

	assign := strings.Index(src, "= [")
	if assign == -1 {
		t.Fatal("launch manifest assignment not found")
	}

	start := strings.Index(src[assign:], "[")
	end := strings.Index(src[assign:], "] satisfies")
	if end != -1 {
		end += assign
	} else {
		end = strings.LastIndex(src, "]")
	}
	if start == -1 || end == -1 || end < start {
		t.Fatal("launch manifest array not found")
	}

	start += assign

	return src[start : end+1]
}

func TestAppLaunchManifestMatchesLauncherRegistry(t *testing.T) {
	manifest := loadAppLaunchManifest(t)
	infos := ListIntegrationInfos()

	if len(manifest) != len(infos) {
		t.Fatalf("manifest integration count = %d, want %d", len(manifest), len(infos))
	}

	for i, info := range infos {
		entry := manifest[i]

		if entry.ID != info.Name {
			t.Fatalf("manifest[%d].id = %q, want %q", i, entry.ID, info.Name)
		}
		if entry.Name != info.DisplayName {
			t.Fatalf("manifest[%d].name = %q, want %q", i, entry.Name, info.DisplayName)
		}
		if entry.Description != info.Description {
			t.Fatalf("manifest[%d].description = %q, want %q", i, entry.Description, info.Description)
		}

		wantCommand := "ollama launch " + info.Name
		if entry.Command != wantCommand {
			t.Fatalf("manifest[%d].command = %q, want %q", i, entry.Command, wantCommand)
		}
	}
}
