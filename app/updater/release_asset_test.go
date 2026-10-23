package updater

import (
	"encoding/json"
	"runtime"
	"strings"
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

func withSignatures(assets []testAsset) []testAsset {
	out := append([]testAsset{}, assets...)
	for _, a := range assets {
		lower := strings.ToLower(a.Name)
		if strings.HasSuffix(lower, ".zip") || strings.HasSuffix(lower, ".exe") {
			out = append(out, testAsset{Name: a.Name + ".ed25519", URL: a.URL + ".ed25519"})
		}
	}
	return out
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
		name    string
		assets  []testAsset
		want    string
		wantSig string
		signed  bool
	}{
		{"dmg only", []testAsset{{"Ollama-0.36.0-b1-x-darwin-arm64.dmg", "u/dmg"}}, "", "", false},
		{"stable dmg only", []testAsset{{"Ollama.dmg", "u/stable-dmg"}}, "", "", false},
		{"stable dmg beside universal zip", []testAsset{{"Ollama.dmg", "u/stable-dmg"}, {universal, "u/universal"}}, "u/universal", "u/universal.ed25519", true},
		{"stable dmg does not override own arch", []testAsset{{"Ollama.dmg", "u/stable-dmg"}, {universal, "u/universal"}, {own, "u/own"}}, "u/own", "u/own.ed25519", true},
		{"release notes and checksum ignored", []testAsset{{"sha256sum.txt", "u/sum"}, {"Ollama.dmg", "u/stable-dmg"}, {own, "u/own"}}, "u/own", "u/own.ed25519", true},
		{"own arch wins", []testAsset{{other, "u/other"}, {universal, "u/universal"}, {own, "u/own"}}, "u/own", "u/own.ed25519", true},
		{"universal fallback", []testAsset{{other, "u/other"}, {universal, "u/universal"}}, "u/universal", "u/universal.ed25519", true},
		{"unsigned own arch falls through to signed universal", []testAsset{{own, "u/own"}, {universal, "u/universal"}}, "u/universal", "u/universal.ed25519", false},
		{"never other arch", []testAsset{{other, "u/other"}}, "", "", true},
		{"unsigned zip is not installed", []testAsset{{universal, "u/universal"}}, "", "", false},
		{"stock name", []testAsset{{"Ollama-darwin" + ext, "u/stock"}}, "u/stock", "u/stock.ed25519", true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			assets := tt.assets
			if tt.signed {
				assets = withSignatures(assets)
			} else if tt.want != "" {
				// Sign every installer except the unsigned own-arch asset.
				signed := make([]testAsset, 0, len(assets))
				for _, a := range assets {
					if a.URL == "u/own" {
						signed = append(signed, a)
						continue
					}
					signed = append(signed, withSignatures([]testAsset{a})...)
				}
				assets = signed
			}
			got, sig, rel, err := githubReleaseAsset(releaseJSON(t, "v0.36.0-b1-x", time.Unix(0, 0), assets...))
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Errorf("asset = %q, want %q", got, tt.want)
			}
			if sig != tt.wantSig {
				t.Errorf("signature = %q, want %q", sig, tt.wantSig)
			}
			if rel.TagName != "v0.36.0-b1-x" {
				t.Errorf("tag = %q", rel.TagName)
			}
		})
	}
}
