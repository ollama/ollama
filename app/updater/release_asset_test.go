package updater

import (
	"encoding/json"
	"runtime"
	"testing"
	"time"
)

type testAsset struct {
	Name string `json:"name"`
	URL  string `json:"browser_download_url"`
}

func releaseJSON(t *testing.T, tag string, published time.Time, assets ...testAsset) []byte {
	t.Helper()
	b, err := json.Marshal(map[string]any{
		"tag_name":     tag,
		"published_at": published.Format(time.RFC3339),
		"assets":       assets,
	})
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func otherArch() string {
	if runtime.GOARCH == "arm64" {
		return "amd64"
	}
	return "arm64"
}

func TestGitHubReleaseAssetSelection(t *testing.T) {
	ext := ".zip"
	if runtime.GOOS == "windows" {
		ext = ".exe"
	}
	own := "Ollama-0.36.0-b1-x-" + runtime.GOOS + "-" + runtime.GOARCH + ext
	other := "Ollama-0.36.0-b1-x-" + runtime.GOOS + "-" + otherArch() + ext
	universal := "Ollama-0.36.0-b1-x-" + runtime.GOOS + "-universal" + ext

	for _, tt := range []struct {
		name   string
		assets []testAsset
		want   string
	}{
		{"dmg only", []testAsset{{"Ollama-0.36.0-b1-x-darwin-arm64.dmg", "u/dmg"}}, ""},
		{"stable dmg only", []testAsset{{"Ollama.dmg", "u/stable-dmg"}}, ""},
		{"stable dmg beside universal zip", []testAsset{{"Ollama.dmg", "u/stable-dmg"}, {universal, "u/universal"}}, "u/universal"},
		{"stable dmg does not override own arch", []testAsset{{"Ollama.dmg", "u/stable-dmg"}, {universal, "u/universal"}, {own, "u/own"}}, "u/own"},
		{"release notes and checksum ignored", []testAsset{{"sha256sum.txt", "u/sum"}, {"Ollama.dmg", "u/stable-dmg"}, {own, "u/own"}}, "u/own"},
		{"own arch wins", []testAsset{{other, "u/other"}, {universal, "u/universal"}, {own, "u/own"}}, "u/own"},
		{"universal fallback", []testAsset{{other, "u/other"}, {universal, "u/universal"}}, "u/universal"},
		{"never other arch", []testAsset{{other, "u/other"}}, ""},
		{"stock name", []testAsset{{"Ollama-darwin" + ext, "u/stock"}}, "u/stock"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, rel, err := githubReleaseAsset(releaseJSON(t, "v0.36.0-b1-x", time.Unix(0, 0), tt.assets...))
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Errorf("asset = %q, want %q", got, tt.want)
			}
			if rel.TagName != "v0.36.0-b1-x" {
				t.Errorf("tag = %q", rel.TagName)
			}
		})
	}
}
