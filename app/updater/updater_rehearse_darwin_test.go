//go:build updater_rehearse

package updater

import (
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/app/updatetrust"
	"github.com/ollama/ollama/app/version"
)

// TestMacUpdateRehearsal installs a signed stub build over an older signed
// stub and rejects a tampered zip. Run it through scripts/e2e-update-mac.sh.
// It uses a temporary directory, never /Applications.
func TestMacUpdateRehearsal(t *testing.T) {
	signID := developerID(t)
	pub, priv, err := updatetrust.GenerateKeyPair()
	if err != nil {
		t.Fatal(err)
	}

	root := t.TempDir()
	oldBundle, oldStage, oldBackup, oldMarker := BundlePath, UpdateStageDir, appBackupDir, UpgradeMarkerFile
	oldKey := updatePublicKey
	oldURL, oldVersion, oldBuildTime := UpdateCheckURLBase, version.Version, currentBuildTime
	t.Cleanup(func() {
		BundlePath, UpdateStageDir, appBackupDir, UpgradeMarkerFile = oldBundle, oldStage, oldBackup, oldMarker
		updatePublicKey = oldKey
		UpdateCheckURLBase, version.Version, currentBuildTime = oldURL, oldVersion, oldBuildTime
	})

	updatePublicKey = pub
	BundlePath = filepath.Join(root, "Installed.app")
	UpdateStageDir = filepath.Join(root, "updates")
	appBackupDir = filepath.Join(root, "backup")
	UpgradeMarkerFile = filepath.Join(root, "upgraded")
	version.Version = "0.40.5-b11724-ds41.900-0-gabcdef0"
	installedAt := time.Now()
	currentBuildTime = func() time.Time { return installedAt }

	const oldVer = "0.40.5-b11724-ds41.900"
	const newVer = "0.40.5-b11724-ds41.901"
	buildStubApp(t, BundlePath, oldVer, signID)
	if got := plistVersion(t, BundlePath); got != oldVer {
		t.Fatalf("installed version = %s", got)
	}

	newApp := filepath.Join(root, "New.app")
	buildStubApp(t, newApp, newVer, signID)
	zipPath := filepath.Join(root, "new.zip")
	ditto := exec.Command("ditto", "-c", "-k", "--norsrc", "--keepParent", newApp, zipPath)
	if out, err := ditto.CombinedOutput(); err != nil {
		t.Fatalf("ditto: %v\n%s", err, out)
	}
	if err := updatetrust.WriteSignature(priv, zipPath); err != nil {
		t.Fatal(err)
	}
	zipBytes, err := os.ReadFile(zipPath)
	if err != nil {
		t.Fatal(err)
	}
	sigBytes, err := os.ReadFile(zipPath + updatetrust.SignatureSuffix)
	if err != nil {
		t.Fatal(err)
	}

	zipName := "Ollama-" + newVer + "-darwin-universal.zip"
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/repos/maxkongerskov/ollama/releases/latest":
			body := releaseJSON(t, "v"+newVer, installedAt.Add(-time.Hour),
				testAsset{Name: zipName, URL: server.URL + "/dl/" + zipName},
				testAsset{Name: zipName + ".ed25519", URL: server.URL + "/dl/" + zipName + ".ed25519"},
				testAsset{Name: "Ollama.dmg", URL: server.URL + "/dl/Ollama.dmg"},
			)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write(body)
		case "/dl/" + zipName:
			w.Header().Set("ETag", `"rehearsal"`)
			if r.Method == http.MethodGet {
				_, _ = w.Write(zipBytes)
			}
		case "/dl/" + zipName + ".ed25519":
			if r.Method == http.MethodGet {
				_, _ = w.Write(sigBytes)
			}
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL + "/repos/maxkongerskov/ollama/releases/latest"

	ok, resp := (&Updater{}).checkForUpdate(t.Context())
	if !ok {
		t.Fatal("newer signed release was not offered")
	}
	if resp.UpdateVersion != newVer || resp.SignatureURL == "" {
		t.Fatalf("update response = %+v", resp)
	}
	if err := (&Updater{}).DownloadNewRelease(t.Context(), resp); err != nil {
		t.Fatalf("download: %v", err)
	}
	if got := plistVersion(t, BundlePath); got != oldVer {
		t.Fatalf("download changed the installed app to %s", got)
	}

	staged := getStagedUpdate()
	if staged == "" {
		t.Fatal("update was not staged")
	}
	original, err := os.ReadFile(staged)
	if err != nil {
		t.Fatal(err)
	}
	tampered := append([]byte{}, original...)
	tampered[len(tampered)-1] ^= 0xff
	if err := os.WriteFile(staged, tampered, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := VerifyDownload(); err == nil {
		t.Fatal("tampered zip was accepted")
	}
	if got := plistVersion(t, BundlePath); got != oldVer {
		t.Fatalf("rejected update changed the installed app to %s", got)
	}

	if err := os.WriteFile(staged, original, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := VerifyDownload(); err != nil {
		t.Fatalf("restored zip was rejected: %v", err)
	}
	if err := DoUpgrade(false); err != nil {
		t.Fatalf("upgrade: %v", err)
	}
	if got := plistVersion(t, BundlePath); got != newVer {
		t.Fatalf("installed version after upgrade = %s, want %s", got, newVer)
	}
}

func developerID(t *testing.T) string {
	t.Helper()
	if id := strings.TrimSpace(os.Getenv("OLLAMA_SIGN_ID")); id != "" {
		return id
	}
	out, err := exec.Command("security", "find-identity", "-v", "-p", "codesigning").Output()
	if err != nil {
		t.Fatalf("security find-identity: %v", err)
	}
	for _, line := range strings.Split(string(out), "\n") {
		if strings.Contains(line, "Developer ID Application") && strings.Contains(line, "6LZ2DS9JPD") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				return fields[1]
			}
		}
	}
	t.Fatal("No Developer ID Application identity for team 6LZ2DS9JPD")
	return ""
}

func buildStubApp(t *testing.T, appPath, appVersion, signID string) {
	t.Helper()
	macOS := filepath.Join(appPath, "Contents", "MacOS")
	if err := os.MkdirAll(macOS, 0o755); err != nil {
		t.Fatal(err)
	}
	arch := "arm64"
	if runtime.GOARCH == "amd64" {
		arch = "x86_64"
	}
	src := filepath.Join(t.TempDir(), "stub.c")
	if err := os.WriteFile(src, []byte("int main(void){return 0;}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(macOS, "Ollama")
	clang := exec.Command("clang", "-arch", arch, "-o", bin, src)
	if out, err := clang.CombinedOutput(); err != nil {
		t.Fatalf("clang: %v\n%s", err, out)
	}
	plist := filepath.Join(appPath, "Contents", "Info.plist")
	body := `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>Ollama</string>
<key>CFBundleIdentifier</key><string>ai.ollama.ollama.e2e</string>
<key>CFBundleName</key><string>Ollama</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>` + appVersion + `</string>
<key>CFBundleVersion</key><string>` + appVersion + `</string>
</dict></plist>
`
	if err := os.WriteFile(plist, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	sign := exec.Command("codesign", "--force", "--timestamp", "--sign", signID, "--identifier", "ai.ollama.ollama.e2e", "--options", "runtime", appPath)
	if out, err := sign.CombinedOutput(); err != nil {
		t.Fatalf("codesign: %v\n%s", err, out)
	}
	info, err := exec.Command("codesign", "-dv", appPath).CombinedOutput()
	if err != nil {
		t.Fatalf("codesign -dv: %v\n%s", err, info)
	}
	if !strings.Contains(string(info), "TeamIdentifier=6LZ2DS9JPD") {
		t.Fatalf("stub app is not team 6LZ2DS9JPD:\n%s", info)
	}
}

func plistVersion(t *testing.T, appPath string) string {
	t.Helper()
	out, err := exec.Command("plutil", "-extract", "CFBundleShortVersionString", "raw", filepath.Join(appPath, "Contents", "Info.plist")).Output()
	if err != nil {
		t.Fatalf("plutil: %v", err)
	}
	return strings.TrimSpace(string(out))
}
