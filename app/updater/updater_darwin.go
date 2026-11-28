package updater

// #cgo CFLAGS: -x objective-c
// #cgo LDFLAGS: -framework Webkit -framework Cocoa -framework LocalAuthentication -framework ServiceManagement
// #include "updater_darwin.h"
// typedef const char cchar_t;
import "C"

import (
	"archive/zip"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"
)

const updateArchiveRoot = "Ollama.app"

type bundleEntryScope int

const (
	bundleEntryRelative bundleEntryScope = iota
	bundleEntryWithArchiveRoot
)

var (
	appBackupDir   string
	SystemWidePath = "/Applications/Ollama.app"
)

var BundlePath = func() string {
	if bundle := alreadyMoved(); bundle != "" {
		return bundle
	}

	exe, err := os.Executable()
	if err != nil {
		return ""
	}

	// We also install this binary in Contents/Frameworks/Squirrel.framework/Versions/A/Squirrel
	if filepath.Base(exe) == "Squirrel" &&
		filepath.Base(filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(exe)))))) == "Contents" {
		return filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(exe))))))
	}

	// Make sure we're in a proper macOS app bundle structure (Contents/MacOS)
	if filepath.Base(filepath.Dir(exe)) != "MacOS" ||
		filepath.Base(filepath.Dir(filepath.Dir(exe))) != "Contents" {
		return ""
	}

	return filepath.Dir(filepath.Dir(filepath.Dir(exe)))
}()

func init() {
	VerifyDownload = verifyDownload
	Installer = "Ollama-darwin.zip"
	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}

	var uts unix.Utsname
	if err := unix.Uname(&uts); err == nil {
		sysname := unix.ByteSliceToString(uts.Sysname[:])
		release := unix.ByteSliceToString(uts.Release[:])
		UserAgentOS = fmt.Sprintf("%s/%s", sysname, release)
	} else {
		slog.Warn("unable to determine OS version", "error", err)
		UserAgentOS = "Darwin"
	}

	// TODO handle failure modes here, and developer mode better...

	// Executable = Ollama.app/Contents/MacOS/Ollama

	UpgradeLogFile = filepath.Join(home, ".ollama", "logs", "upgrade.log")

	cacheDir, err := os.UserCacheDir()
	if err != nil {
		slog.Warn("unable to determine user cache dir, falling back to tmpdir", "error", err)
		cacheDir = os.TempDir()
	}
	appDataDir := filepath.Join(cacheDir, "ollama")
	UpgradeMarkerFile = filepath.Join(appDataDir, "upgraded")
	appBackupDir = filepath.Join(appDataDir, "backup")
	UpdateStageDir = filepath.Join(appDataDir, "updates")
}

func DoUpgrade(interactive bool) error {
	// TODO use UpgradeLogFile to record the upgrade details from->to version, etc.

	bundle := getStagedUpdate()
	if bundle == "" {
		return fmt.Errorf("failed to lookup downloads")
	}

	slog.Info("starting upgrade", "app", BundlePath, "update", bundle, "pid", os.Getpid(), "log", UpgradeLogFile)

	// TODO - in the future, consider shutting down the backend server now to give it
	// time to drain connections and stop allowing new connections while we perform the
	// actual upgrade to reduce the overall time to complete
	contentsName := filepath.Join(BundlePath, "Contents")
	appBackup := filepath.Join(appBackupDir, "Ollama.app")
	contentsOldName := filepath.Join(appBackup, "Contents")

	// Verify old doesn't exist yet
	if _, err := os.Stat(contentsOldName); err == nil {
		slog.Error("prior upgrade failed", "backup", contentsOldName)
		return fmt.Errorf("prior upgrade failed - please upgrade manually by installing the bundle")
	}
	if err := os.MkdirAll(appBackupDir, 0o755); err != nil {
		return fmt.Errorf("unable to create backup dir %s: %w", appBackupDir, err)
	}

	// Verify bundle loads before starting staging process
	r, err := zip.OpenReader(bundle)
	if err != nil {
		return fmt.Errorf("unable to open upgrade bundle %s: %w", bundle, err)
	}
	defer r.Close()

	slog.Debug("temporarily staging old version", "staging", appBackup)
	if err := os.Rename(BundlePath, appBackup); err != nil {
		if !interactive {
			// We don't want to prompt for permission if we're attempting to upgrade at startup
			return fmt.Errorf("unable to upgrade in non-interactive mode with permission problems: %w", err)
		}
		slog.Warn("unable to backup old version due to permission problems, using privileged update", "error", err)
		return upgradeWithAuthorization(r)
	}

	// Get ready to try to unwind a partial upgrade failure during unzip
	// If something goes wrong, we attempt to put the old version back.
	anyFailures := false
	defer func() {
		if anyFailures {
			slog.Warn("upgrade failures detected, attempting to revert")
			if err := os.RemoveAll(BundlePath); err != nil {
				slog.Warn("failed to remove partial upgrade", "path", BundlePath, "error", err)
				// At this point, we're basically hosed and the user will need to re-install
				return
			}
			if err := os.Rename(appBackup, BundlePath); err != nil {
				slog.Error("failed to revert to prior version", "path", contentsName, "error", err)
			}
		}
	}()

	if err := extractBundle(r, BundlePath); err != nil {
		anyFailures = true
		return err
	}

	finishUpgrade()
	return nil
}

// upgradeWithAuthorization updates a bundle the current user can't modify. The
// update is extracted as the current user, then a single privileged step swaps
// it into place and preserves the original owner, so the bundle is never
// handed over to the logged-in user.
func upgradeWithAuthorization(r *zip.ReadCloser) error {
	stageRoot := filepath.Join(appBackupDir, "staged")
	if err := os.RemoveAll(stageRoot); err != nil {
		return fmt.Errorf("unable to clean staging dir %s: %w", stageRoot, err)
	}
	defer os.RemoveAll(stageRoot)
	newApp := filepath.Join(stageRoot, "Ollama.app")
	if err := extractBundle(r, newApp); err != nil {
		return err
	}

	fi, err := os.Stat(BundlePath)
	if err != nil {
		return err
	}
	st, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return fmt.Errorf("unable to determine owner of %s", BundlePath)
	}
	owner := fmt.Sprintf("%d:%d", st.Uid, st.Gid)

	if !swapBundle(BundlePath, newApp, owner) {
		return fmt.Errorf("unable to get permission to complete upgrade")
	}
	// The exit status of the privileged step isn't available, so verify.
	if _, err := os.Stat(newApp); err == nil {
		return fmt.Errorf("unable to perform upgrade - privileged update did not complete")
	}
	if _, err := os.Stat(filepath.Join(BundlePath, "Contents")); err != nil {
		return fmt.Errorf("unable to perform upgrade - bundle missing after update: %w", err)
	}

	finishUpgrade()
	return nil
}

func finishUpgrade() {
	f, err := os.OpenFile(UpgradeMarkerFile, os.O_RDONLY|os.O_CREATE, 0o666)
	if err != nil {
		slog.Warn("unable to create marker file", "file", UpgradeMarkerFile, "error", err)
	} else {
		f.Close()
	}
	// Make sure to remove the staged download now that we succeeded so we don't inadvertently try again.
	cleanupOldDownloads(UpdateStageDir)
}

// extractBundle extracts the Ollama.app/ contents of the archive into root.
func extractBundle(r *zip.ReadCloser, root string) error {
	// Bundle contents Ollama.app/Contents/...
	links := []*zip.File{}
	for _, f := range r.File {
		s := strings.SplitN(f.Name, "/", 2)
		if len(s) < 2 || s[1] == "" {
			slog.Debug("skipping", "file", f.Name)
			continue
		}
		name := s[1]
		if strings.HasSuffix(name, "/") {
			d, err := bundleEntryPath(root, name, bundleEntryRelative)
			if err != nil {
				return err
			}
			err = os.MkdirAll(d, 0o755)
			if err != nil {
				return fmt.Errorf("failed to mkdir %s: %w", d, err)
			}
			continue
		}
		if f.Mode()&os.ModeSymlink != 0 {
			// Defer links to the end
			links = append(links, f)
			continue
		}

		destName, err := bundleEntryPath(root, name, bundleEntryRelative)
		if err != nil {
			return err
		}
		if err := extractBundleFile(f, destName, name); err != nil {
			return err
		}
	}
	for _, f := range links {
		s := strings.SplitN(f.Name, "/", 2) // Strip off Ollama.app/
		if len(s) < 2 || s[1] == "" {
			slog.Debug("skipping link", "file", f.Name)
			continue
		}
		name := s[1]
		src, err := f.Open()
		if err != nil {
			return err
		}
		buf, err := io.ReadAll(src)
		if err != nil {
			return err
		}
		link := string(buf)
		if link == "" {
			return fmt.Errorf("bundle contains empty symlink %s", f.Name)
		}
		if filepath.IsAbs(link) {
			return fmt.Errorf("bundle contains absolute symlink %s -> %s", f.Name, link)
		}
		if !validBundleLinkTarget(name, link, bundleEntryRelative) {
			return fmt.Errorf("bundle contains invalid symlink %s -> %s", f.Name, link)
		}
		destName, err := bundleEntryPath(root, name, bundleEntryRelative)
		if err != nil {
			return err
		}
		if err = os.Symlink(link, destName); err != nil {
			return err
		}
	}

	return nil
}

func DoPostUpgradeCleanup() error {
	slog.Debug("post upgrade cleanup", "backup", appBackupDir)
	err := os.RemoveAll(appBackupDir)
	if err != nil {
		return err
	}
	slog.Debug("post upgrade cleanup", "old", UpgradeMarkerFile)
	return os.Remove(UpgradeMarkerFile)
}

func verifyDownload() error {
	bundle := getStagedUpdate()
	if bundle == "" {
		return fmt.Errorf("failed to lookup downloads")
	}
	slog.Debug("verifying update", "bundle", bundle)

	// Extract zip file into a temporary location so we can run the cert verification routines
	dir, err := os.MkdirTemp("", "ollama_update_verify")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	r, err := zip.OpenReader(bundle)
	if err != nil {
		return fmt.Errorf("unable to open upgrade bundle %s: %w", bundle, err)
	}
	defer r.Close()
	links := []*zip.File{}
	for _, f := range r.File {
		if strings.HasSuffix(f.Name, "/") {
			d, err := bundleEntryPath(dir, f.Name, bundleEntryWithArchiveRoot)
			if err != nil {
				return err
			}
			err = os.MkdirAll(d, 0o755)
			if err != nil {
				return fmt.Errorf("failed to mkdir %s: %w", d, err)
			}
			continue
		}
		if f.Mode()&os.ModeSymlink != 0 {
			// Defer links to the end
			links = append(links, f)
			continue
		}
		destName, err := bundleEntryPath(dir, f.Name, bundleEntryWithArchiveRoot)
		if err != nil {
			return err
		}
		if err := extractBundleFile(f, destName, f.Name); err != nil {
			return err
		}
	}
	for _, f := range links {
		src, err := f.Open()
		if err != nil {
			return err
		}
		buf, err := io.ReadAll(src)
		if err != nil {
			return err
		}
		link := string(buf)
		if link == "" {
			return fmt.Errorf("bundle contains empty symlink %s", f.Name)
		}
		if filepath.IsAbs(link) {
			return fmt.Errorf("bundle contains absolute symlink %s -> %s", f.Name, link)
		}
		if !validBundleLinkTarget(f.Name, link, bundleEntryWithArchiveRoot) {
			return fmt.Errorf("bundle contains invalid symlink %s -> %s", f.Name, link)
		}
		destName, err := bundleEntryPath(dir, f.Name, bundleEntryWithArchiveRoot)
		if err != nil {
			return err
		}
		if err = os.Symlink(link, destName); err != nil {
			return err
		}
	}

	if err := verifyExtractedBundle(filepath.Join(dir, "Ollama.app")); err != nil {
		return fmt.Errorf("signature verification failed: %s", err)
	}
	return nil
}

func bundleEntryPath(root, name string, scope bundleEntryScope) (string, error) {
	cleanName := filepath.Clean(filepath.FromSlash(name))
	if !filepath.IsLocal(cleanName) {
		return "", fmt.Errorf("bundle contains invalid path: %s", name)
	}
	if scope == bundleEntryWithArchiveRoot && cleanName != updateArchiveRoot &&
		!strings.HasPrefix(cleanName, updateArchiveRoot+string(os.PathSeparator)) {
		return "", fmt.Errorf("bundle contains invalid path: %s", name)
	}
	return filepath.Join(root, cleanName), nil
}

func extractBundleFile(f *zip.File, destName, name string) error {
	src, err := f.Open()
	if err != nil {
		return fmt.Errorf("failed to open bundle file %s: %w", name, err)
	}
	defer src.Close()

	d := filepath.Dir(destName)
	if _, err := os.Stat(d); err != nil {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return fmt.Errorf("failed to mkdir %s: %w", d, err)
		}
	}

	destFile, err := os.OpenFile(destName, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
	if err != nil {
		return fmt.Errorf("failed to open output file %s: %w", destName, err)
	}
	defer destFile.Close()

	if _, err := io.Copy(destFile, src); err != nil {
		return fmt.Errorf("failed to open extract file %s: %w", destName, err)
	}
	return nil
}

func validBundleLinkTarget(name, link string, scope bundleEntryScope) bool {
	cleanTarget := filepath.Clean(filepath.Join(filepath.Dir(filepath.FromSlash(name)), filepath.FromSlash(link)))
	if !filepath.IsLocal(cleanTarget) {
		return false
	}
	return scope == bundleEntryRelative || cleanTarget == updateArchiveRoot ||
		strings.HasPrefix(cleanTarget, updateArchiveRoot+string(os.PathSeparator))
}

// If we detect an upgrade bundle, attempt to upgrade at startup
func DoUpgradeAtStartup() error {
	bundle := getStagedUpdate()
	if bundle == "" {
		return fmt.Errorf("failed to lookup downloads")
	}

	if BundlePath == "" {
		return fmt.Errorf("unable to upgrade at startup, app in development mode")
	}

	// [Re]verify before proceeding
	if err := VerifyDownload(); err != nil {
		_ = os.Remove(bundle)
		slog.Warn("verification failure", "bundle", bundle, "error", err)
		return nil
	}
	slog.Info("performing update at startup", "bundle", bundle)
	return DoUpgrade(false)
}

func getStagedUpdate() string {
	files, err := filepath.Glob(filepath.Join(UpdateStageDir, "*", "*.zip"))
	if err != nil {
		slog.Debug("failed to lookup downloads", "error", err)
		return ""
	}
	if len(files) == 0 {
		return ""
	} else if len(files) > 1 {
		// Shouldn't happen
		slog.Warn("multiple update downloads found, using first one", "bundles", files)
	}
	return files[0]
}

func IsUpdatePending() bool {
	return getStagedUpdate() != ""
}

// swapBundleScript atomically replaces the app bundle ($1) with a staged
// bundle ($2) and gives it the owner ($3, uid:gid) of the bundle it replaces.
// The old bundle is restored if the swap fails part way.
const swapBundleScript = `set -e
app="$1"; new="$2"; owner="$3"
old="$app.old.$$"
chown -R "$owner" "$new"
mv "$app" "$old"
if ! mv "$new" "$app"; then
	mv "$old" "$app"
	exit 1
fi
rm -rf "$old"
`

// swapBundle replaces app with newApp. It is a variable so tests can run it
// without prompting for administrator credentials.
var swapBundle = swapBundleWithAuthorization

func swapBundleWithAuthorization(app, newApp, owner string) bool {
	script := C.CString(swapBundleScript)
	defer C.free(unsafe.Pointer(script))
	a := C.CString(app)
	defer C.free(unsafe.Pointer(a))
	n := C.CString(newApp)
	defer C.free(unsafe.Pointer(n))
	o := C.CString(owner)
	defer C.free(unsafe.Pointer(o))
	return bool(C.runScriptWithAuthorization(script, a, n, o))
}

func verifyExtractedBundle(path string) error {
	p := C.CString(path)
	defer C.free(unsafe.Pointer(p))
	resp := C.verifyExtractedBundle(p)
	if resp == nil {
		return nil
	}

	return errors.New(C.GoString(resp))
}

//export goLogInfo
func goLogInfo(msg *C.cchar_t) {
	slog.Info(C.GoString(msg))
}

//export goLogDebug
func goLogDebug(msg *C.cchar_t) {
	slog.Debug(C.GoString(msg))
}

func alreadyMoved() string {
	// Respect users intent if they chose "keep" vs. "replace" when dragging to Applications
	installedAppPaths, err := filepath.Glob(filepath.Join(
		strings.TrimSuffix(SystemWidePath, filepath.Ext(SystemWidePath))+"*"+filepath.Ext(SystemWidePath),
		"Contents", "MacOS", "Ollama"))
	if err != nil {
		slog.Warn("failed to lookup installed app paths", "error", err)
		return ""
	}
	exe, err := os.Executable()
	if err != nil {
		slog.Warn("failed to resolve executable", "error", err)
		return ""
	}
	self, err := os.Stat(exe)
	if err != nil {
		slog.Warn("failed to stat running executable", "path", exe, "error", err)
		return ""
	}
	selfSys := self.Sys().(*syscall.Stat_t)
	for _, installedAppPath := range installedAppPaths {
		app, err := os.Stat(installedAppPath)
		if err != nil {
			slog.Debug("failed to stat installed app path", "path", installedAppPath, "error", err)
			continue
		}
		appSys := app.Sys().(*syscall.Stat_t)

		if appSys.Ino == selfSys.Ino {
			return filepath.Dir(filepath.Dir(filepath.Dir(installedAppPath)))
		}
	}
	return ""
}
