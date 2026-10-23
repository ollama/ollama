// Asset selection lives outside the windows/darwin build tag so it can be
// tested on Linux. The desktop updater (updater.go) is still built only for
// Windows and macOS. Disk images, including the stable Ollama.dmg download,
// are not installable here; the app stages a .zip or .exe.
package updater

import (
	"encoding/json"
	"runtime"
	"strings"
	"time"
)

// githubReleaseResponse is the subset of a GitHub release the updater reads.
type githubReleaseResponse struct {
	TagName     string    `json:"tag_name"`
	PublishedAt time.Time `json:"published_at"`
	Assets      []struct {
		Name               string `json:"name"`
		BrowserDownloadURL string `json:"browser_download_url"`
	} `json:"assets"`
}

// githubReleaseAsset picks the installable asset from a fork release: a .zip
// on macOS or an .exe on Windows, plus the sibling "<name>.ed25519" signature.
// An asset naming this OS and arch wins; otherwise the first asset that does
// not name another arch is used. A zip or exe with no signature asset is
// skipped, so an unsigned arch-specific build falls through to a signed
// universal zip.
//
// Disk images are ignored, including the stable Ollama.dmg asset the README
// links at /releases/latest/download/Ollama.dmg. The updater stages the
// versioned zip (or the Windows exe) and cannot install a DMG.
func githubReleaseAsset(body []byte) (assetURL, signatureURL string, rel githubReleaseResponse, err error) {
	if err := json.Unmarshal(body, &rel); err != nil {
		return "", "", rel, err
	}
	ext := ".zip"
	if runtime.GOOS == "windows" {
		ext = ".exe"
	}
	signatures := map[string]string{}
	for _, asset := range rel.Assets {
		name := strings.ToLower(asset.Name)
		if strings.HasSuffix(name, ".ed25519") && asset.BrowserDownloadURL != "" {
			signatures[name] = asset.BrowserDownloadURL
		}
	}
	var fallback, fallbackSig string
	for _, asset := range rel.Assets {
		name := strings.ToLower(asset.Name)
		if !strings.HasSuffix(name, ext) || asset.BrowserDownloadURL == "" {
			continue
		}
		sig := signatures[name+".ed25519"]
		if sig == "" {
			continue
		}
		if strings.Contains(name, runtime.GOOS) && assetNamesArch(name, runtime.GOARCH) {
			return asset.BrowserDownloadURL, sig, rel, nil
		}
		if fallback == "" && !assetNamesOtherArch(name, runtime.GOARCH) {
			fallback = asset.BrowserDownloadURL
			fallbackSig = sig
		}
	}
	return fallback, fallbackSig, rel, nil
}

var archAliases = map[string][]string{
	"amd64": {"amd64", "x86_64"},
	"arm64": {"arm64", "aarch64"},
}

func assetNamesArch(name, arch string) bool {
	aliases, ok := archAliases[arch]
	if !ok {
		aliases = []string{arch}
	}
	for _, a := range aliases {
		if strings.Contains(name, a) {
			return true
		}
	}
	return false
}

func assetNamesOtherArch(name, arch string) bool {
	for other := range archAliases {
		if other != arch && assetNamesArch(name, other) {
			return true
		}
	}
	return false
}
