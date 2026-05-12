//go:build windows || darwin

package main

// #cgo CFLAGS: -x objective-c
// #cgo LDFLAGS: -framework Webkit -framework Cocoa -framework LocalAuthentication -framework ServiceManagement
// #include "app_darwin.h"
// #include "../../updater/updater_darwin.h"
// typedef const char cchar_t;
import "C"

import (
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
	"unsafe"

	"github.com/ollama/ollama/app/updater"
	"github.com/ollama/ollama/app/version"
)

var ollamaPath = func() string {
	if updater.BundlePath != "" {
		return filepath.Join(updater.BundlePath, "Contents", "Resources", "ollama")
	}

	pwd, err := os.Getwd()
	if err != nil {
		slog.Warn("failed to get pwd", "error", err)
		return ""
	}
	return filepath.Join(pwd, "ollama")
}()

var (
	isApp           = updater.BundlePath != ""
	appLogPath      = filepath.Join(os.Getenv("HOME"), ".ollama", "logs", "app.log")
	launchAgentPath = filepath.Join(os.Getenv("HOME"), "Library", "LaunchAgents", "com.ollama.ollama.plist")
)

// TODO(jmorganca): pre-create the window and pass
// it to the webview instead of using the internal one
//
//export StartUI
func StartUI(path *C.cchar_t) {
	p := C.GoString(path)
	wv.Run(p)
	styleWindow(wv.webview.Window())
	C.setWindowDelegate(wv.webview.Window())
}

//export ShowUI
func ShowUI() {
	// If webview is already running, just show the window
	if wv.IsRunning() && wv.webview != nil {
		showWindow(wv.webview.Window())
	} else {
		root := C.CString("/")
		defer C.free(unsafe.Pointer(root))
		StartUI(root)
	}
}

//export StopUI
func StopUI() {
	wv.Terminate()
}

//export StartUpdate
func StartUpdate() {
	if err := updater.DoUpgrade(true); err != nil {
		slog.Error("upgrade failed", "error", err)
		return
	}
	slog.Debug("launching new version...")
	// TODO - consider a timer that aborts if this takes too long and we haven't been killed yet...
	LaunchNewApp()
	// not reached if upgrade works, the new app will kill this process
}

//export darwinStartHiddenTasks
func darwinStartHiddenTasks() {
	startHiddenTasks()
}

func init() {
	// Temporary code to mimic Squirrel ShipIt behavior
	if len(os.Args) > 2 {
		if os.Args[1] == "___launch___" {
			path := strings.TrimPrefix(os.Args[2], "file://")
			slog.Info("Ollama binary called as ShipIt - launching", "app", path)
			appName := C.CString(path)
			defer C.free(unsafe.Pointer(appName))
			C.launchApp(appName)
			slog.Info("other instance has been launched")
			time.Sleep(5 * time.Second)
			slog.Info("exiting with zero status")
			os.Exit(0)
		}
	}
}

// maybeMoveAndRestart checks if we should relocate
// and returns true if we did and should immediately exit
func maybeMoveAndRestart() appMove {
	if updater.BundlePath == "" {
		// Typically developer mode with 'go run ./cmd/app'
		return CannotMove
	}
	// Respect users intent if they chose "keep" vs. "replace" when dragging to Applications
	if strings.HasPrefix(updater.BundlePath, strings.TrimSuffix(updater.SystemWidePath, filepath.Ext(updater.SystemWidePath))) {
		return AlreadyMoved
	}

	// Ask to move to applications directory
	status := (appMove)(C.askToMoveToApplications())
	if status == MoveCompleted {
		// Double check
		if _, err := os.Stat(updater.SystemWidePath); err != nil {
			slog.Warn("stat failure after move", "path", updater.SystemWidePath, "error", err)
			return MoveError
		}
	}
	return status
}

// handleExistingInstance handles existing instances on macOS
func handleExistingInstance(_ bool) {
	C.killOtherInstances()
}

func installSymlink() {
	if !isApp {
		return
	}
	cliPath := C.CString(ollamaPath)
	defer C.free(unsafe.Pointer(cliPath))

	// Check the users path first
	cmd, _ := exec.LookPath("ollama")
	if cmd != "" {
		resolved, err := os.Readlink(cmd)
		if err == nil {
			tmp, err := filepath.Abs(resolved)
			if err == nil {
				resolved = tmp
			}
		} else {
			resolved = cmd
		}
		if resolved == ollamaPath {
			slog.Info("ollama already in users PATH", "cli", cmd)
			return
		}
	}

	code := C.installSymlink(cliPath)
	if code != 0 {
		slog.Error("Failed to install symlink")
	}
}

func UpdateAvailable(ver string) error {
	slog.Debug("update detected, adjusting menu")
	// TODO (jmorganca): find a better check for development mode than checking the bundle path
	if updater.BundlePath != "" {
		C.updateAvailable()
	}
	return nil
}

func osRun(_ func(), hasCompletedFirstRun, startHidden bool) {
	registerLaunchAgent(hasCompletedFirstRun)

	// Run the native macOS app
	// Note: this will block until the app is closed
	slog.Debug("starting native darwin event loop")
	C.run(C._Bool(hasCompletedFirstRun), C._Bool(startHidden))
}

func quit() {
	C.quit()
}

func LaunchNewApp() {
	appName := C.CString(updater.BundlePath)
	defer C.free(unsafe.Pointer(appName))
	C.launchApp(appName)
}

func registerLaunchAgent(hasCompletedFirstRun bool) {
	// Remove any stale Login Item registrations
	C.unregisterSelfFromLoginItem()

	C.registerSelfAsLoginItem(C._Bool(hasCompletedFirstRun))
}

func logStartup() {
	appPath := updater.BundlePath
	if appPath == updater.SystemWidePath {
		// Detect sandboxed scenario
		exe, err := os.Executable()
		if err == nil {
			p := filepath.Dir(exe)
			if filepath.Base(p) == "MacOS" {
				p = filepath.Dir(filepath.Dir(p))
				if p != appPath {
					slog.Info("starting sandboxed Ollama", "app", appPath, "sandbox", p)
					return
				}
			}
		}
	}
	slog.Info("starting Ollama", "app", appPath, "version", version.Version, "OS", updater.UserAgentOS)
}

func hideWindow(ptr unsafe.Pointer) {
	C.hideWindow(C.uintptr_t(uintptr(ptr)))
}

func showWindow(ptr unsafe.Pointer) {
	C.showWindow(C.uintptr_t(uintptr(ptr)))
}

func styleWindow(ptr unsafe.Pointer) {
	C.styleWindow(C.uintptr_t(uintptr(ptr)))
}

func runInBackground() {
	cmd := exec.Command(filepath.Join(updater.BundlePath, "Contents", "MacOS", "Ollama"), "hidden")
	if cmd != nil {
		err := cmd.Run()
		if err != nil {
			slog.Error("failed to run Ollama", "bundlePath", updater.BundlePath, "error", err)
			os.Exit(1)
		}
	} else {
		slog.Error("failed to start Ollama in background", "bundlePath", updater.BundlePath)
		os.Exit(1)
	}
}

func drag(ptr unsafe.Pointer) {
	C.drag(C.uintptr_t(uintptr(ptr)))
}

func doubleClick(ptr unsafe.Pointer) {
	C.doubleClick(C.uintptr_t(uintptr(ptr)))
}

//export handleConnectURL
func handleConnectURL() {
	handleConnectURLScheme()
}

// checkAndHandleExistingInstance is not needed on non-Windows platforms
func checkAndHandleExistingInstance(_ string) bool {
	return false
}
