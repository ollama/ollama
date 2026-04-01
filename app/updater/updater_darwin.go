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
		// TODO actually inspect the error and look for permission problems before trying chown
		slog.Warn("unable to backup old version due to permission problems, changing ownership", "error", err)
		u, err := user.Current()
		if err != nil {
			return err
		}
		if !chownWithAuthorization(u.Username) {
			return fmt.Errorf("unable to change permissions to complete upgrade")
		}
		if err := os.Rename(BundlePath, appBackup); err != nil {
			return fmt.Errorf("unable to perform upgrade - failed to stage old version: %w", err)
		}
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
			if err := os.Rename(appBackup, BundlePath); err != nil {
				slog.Error("failed to revert to prior version", "path", contentsName, "error", err)
			}
		}
	}()

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
			d := filepath.Join(BundlePath, name)
			err := os.MkdirAll(d, 0o755)
			if err != nil {
				anyFailures = true
				return fmt.Errorf("failed to mkdir %s: %w", d, err)
			}
			continue
		}
		if f.Mode()&os.ModeSymlink != 0 {
			// Defer links to the end
			links = append(links, f)
			continue
		}

		src, err := f.Open()
		if err != nil {
			anyFailures = true
			return fmt.Errorf("failed to open bundle file %s: %w", name, err)
		}
		destName := filepath.Join(BundlePath, name)
		// Verify directory first
		d := filepath.Dir(destName)
		if _, err := os.Stat(d); err != nil {
			err := os.MkdirAll(d, 0o755)
			if err != nil {
				anyFailures = true
				return fmt.Errorf("failed to mkdir %s: %w", d, err)
			}
		}
		destFile, err := os.OpenFile(destName, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
		if err != nil {
			anyFailures = true
			return fmt.Errorf("failed to open output file %s: %w", destName, err)
		}
		defer destFile.Close()
		if _, err := io.Copy(destFile, src); err != nil {
			anyFailures = true
			return fmt.Errorf("failed to open extract file %s: %w", destName, err)
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
			anyFailures = true
			return err
		}
		buf, err := io.ReadAll(src)
		if err != nil {
			anyFailures = true
			return err
		}
		link := string(buf)
		if link[0] == '/' {
			anyFailures = true
			return fmt.Errorf("bundle contains absolute symlink %s -> %s", f.Name, link)
		}
		// Don't allow links outside of Ollama.app
		if strings.HasPrefix(filepath.Join(filepath.Dir(name), link), "..") {
			anyFailures = true
			return fmt.Errorf("bundle contains link outside of contents %s -> %s", f.Name, link)
		}
		if err = os.Symlink(link, filepath.Join(BundlePath, name)); err != nil {
			anyFailures = true
			return err
		}
	}

	f, err := os.OpenFile(UpgradeMarkerFile, os.O_RDONLY|os.O_CREATE, 0o666)
	if err != nil {
		slog.Warn("unable to create marker file", "file", UpgradeMarkerFile, "error", err)
	}
	f.Close()
	// Make sure to remove the staged download now that we succeeded so we don't inadvertently try again.
	cleanupOldDownloads(UpdateStageDir)

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
			d := filepath.Join(dir, f.Name)
			err := os.MkdirAll(d, 0o755)
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
		src, err := f.Open()
		if err != nil {
			return fmt.Errorf("failed to open bundle file %s: %w", f.Name, err)
		}
		destName := filepath.Join(dir, f.Name)
		// Verify directory first
		d := filepath.Dir(destName)
		if _, err := os.Stat(d); err != nil {
			err := os.MkdirAll(d, 0o755)
			if err != nil {
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
		if link[0] == '/' {
			return fmt.Errorf("bundle contains absolute symlink %s -> %s", f.Name, link)
		}
		if strings.HasPrefix(filepath.Join(filepath.Dir(f.Name), link), "..") {
			return fmt.Errorf("bundle contains link outside of contents %s -> %s", f.Name, link)
		}
		if err = os.Symlink(link, filepath.Join(dir, f.Name)); err != nil {
			return err
		}
	}

	if err := verifyExtractedBundle(filepath.Join(dir, "Ollama.app")); err != nil {
		return fmt.Errorf("signature verification failed: %s", err)
	}
	return nil
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
