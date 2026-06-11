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
	"os/user"
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
	appBackupDir                   string
	SystemWidePath                 = "/Applications/Ollama.app"
	renameBundle                   = os.Rename
	replaceBundleWithAuthorization = replaceBundleWithAuthorizationNative
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
	if err := renameBundle(BundlePath, appBackup); err != nil {
		if !interactive {
			// We don't want to prompt for permission if we're attempting to upgrade at startup
			return fmt.Errorf("unable to upgrade in non-interactive mode with permission problems: %w", err)
		}
		if !isPermissionProblem(err) {
			return fmt.Errorf("unable to perform upgrade - failed to stage old version: %w", err)
		}

		slog.Warn("unable to backup old version due to permission problems, requesting authorization", "error", err)
		if err := doUpgradeWithAuthorization(r.File, appBackup); err != nil {
			return err
		}
		completeUpgrade()
		return nil
	}

	// Get ready to try to unwind a partial upgade failure during unzip
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
			if err := renameBundle(appBackup, BundlePath); err != nil {
				slog.Error("failed to revert to prior version", "path", contentsName, "error", err)
			}
		}
	}()

	if err := extractUpdateBundle(r.File, BundlePath, bundleEntryRelative); err != nil {
		anyFailures = true
		return err
	}

	completeUpgrade()
	return nil
}

func completeUpgrade() {
	f, err := os.OpenFile(UpgradeMarkerFile, os.O_RDONLY|os.O_CREATE, 0o666)
	if err != nil {
		slog.Warn("unable to create marker file", "file", UpgradeMarkerFile, "error", err)
	} else if err := f.Close(); err != nil {
		slog.Warn("unable to close marker file", "file", UpgradeMarkerFile, "error", err)
	}
	// Make sure to remove the staged download now that we succeeded so we don't inadvertently try again.
	cleanupOldDownloads(UpdateStageDir)
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
	if err := extractUpdateBundle(r.File, dir, bundleEntryWithArchiveRoot); err != nil {
		return err
	}

	if err := verifyExtractedBundle(filepath.Join(dir, "Ollama.app")); err != nil {
		return fmt.Errorf("signature verification failed: %s", err)
	}
	return nil
}

func doUpgradeWithAuthorization(files []*zip.File, appBackup string) error {
	u, err := user.Current()
	if err != nil {
		return err
	}
	owner := u.Uid
	if owner == "" {
		owner = u.Username
	}

	stagingDir, err := os.MkdirTemp(appBackupDir, "authorized-update-")
	if err != nil {
		return fmt.Errorf("unable to create authorized update staging dir: %w", err)
	}
	defer os.RemoveAll(stagingDir)

	if err := extractUpdateBundle(files, stagingDir, bundleEntryWithArchiveRoot); err != nil {
		return err
	}
	stagedApp := filepath.Join(stagingDir, updateArchiveRoot)
	if _, err := os.Stat(stagedApp); err != nil {
		return fmt.Errorf("staged update is missing app bundle: %w", err)
	}
	if !replaceBundleWithAuthorization(stagedApp, appBackup, BundlePath, owner) {
		return fmt.Errorf("unable to replace app bundle with authorization")
	}
	return nil
}

func extractUpdateBundle(files []*zip.File, root string, scope bundleEntryScope) error {
	links := []*zip.File{}
	for _, f := range files {
		name, ok := bundleArchiveName(f.Name, scope)
		if !ok {
			slog.Debug("skipping", "file", f.Name)
			continue
		}
		if strings.HasSuffix(name, "/") {
			d, err := bundleEntryPath(root, name, scope)
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
			// Defer links until their target files have been extracted.
			links = append(links, f)
			continue
		}

		destName, err := bundleEntryPath(root, name, scope)
		if err != nil {
			return err
		}
		if err := extractBundleFile(f, destName, name); err != nil {
			return err
		}
	}
	for _, f := range links {
		name, ok := bundleArchiveName(f.Name, scope)
		if !ok {
			slog.Debug("skipping link", "file", f.Name)
			continue
		}
		src, err := f.Open()
		if err != nil {
			return err
		}
		buf, err := io.ReadAll(src)
		if closeErr := src.Close(); closeErr != nil && err == nil {
			err = closeErr
		}
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
		if !validBundleLinkTarget(name, link, scope) {
			return fmt.Errorf("bundle contains invalid symlink %s -> %s", f.Name, link)
		}
		destName, err := bundleEntryPath(root, name, scope)
		if err != nil {
			return err
		}
		if err = os.Symlink(link, destName); err != nil {
			return err
		}
	}
	return nil
}

func bundleArchiveName(name string, scope bundleEntryScope) (string, bool) {
	if scope == bundleEntryWithArchiveRoot {
		return name, true
	}

	parts := strings.SplitN(name, "/", 2)
	if len(parts) < 2 || parts[1] == "" {
		return "", false
	}
	return parts[1], true
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

func isPermissionProblem(err error) bool {
	return errors.Is(err, os.ErrPermission) ||
		errors.Is(err, syscall.EACCES) ||
		errors.Is(err, syscall.EPERM)
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

func chownWithAuthorization(user string) bool {
	u := C.CString(user)
	defer C.free(unsafe.Pointer(u))
	return (bool)(C.chownWithAuthorization(u))
}

func replaceBundleWithAuthorizationNative(stagedApp, backupApp, destApp, owner string) bool {
	staged := C.CString(stagedApp)
	defer C.free(unsafe.Pointer(staged))
	backup := C.CString(backupApp)
	defer C.free(unsafe.Pointer(backup))
	dest := C.CString(destApp)
	defer C.free(unsafe.Pointer(dest))
	ownerName := C.CString(owner)
	defer C.free(unsafe.Pointer(ownerName))
	return (bool)(C.replaceBundleWithAuthorization(staged, backup, dest, ownerName))
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
