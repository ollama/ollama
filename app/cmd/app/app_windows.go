//go:build windows || darwin

package main

import (
	"errors"
	"fmt"
	"io"
	"log"
	"log/slog"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"

	"github.com/ollama/ollama/app/updater"
	"github.com/ollama/ollama/app/version"
	"github.com/ollama/ollama/app/wintray"
	"golang.org/x/sys/windows"
)

var (
	appPath         = filepath.Join(os.Getenv("LOCALAPPDATA"), "Programs", "Ollama")
	appLogPath      = filepath.Join(os.Getenv("LOCALAPPDATA"), "Ollama", "app.log")
	startupShortcut = filepath.Join(os.Getenv("APPDATA"), "Microsoft", "Windows", "Start Menu", "Programs", "Startup", "Ollama.lnk")
	ollamaPath      string
	DesktopAppName  = "ollama app.exe"
)

func init() {
	// With alternate install location use executable location
	exe, err := os.Executable()
	if err != nil {
		slog.Warn("error discovering executable directory", "error", err)
	} else {
		appPath = filepath.Dir(exe)
	}
	ollamaPath = filepath.Join(appPath, "ollama.exe")

	// Handle developer mode (go run ./cmd/app)
	if _, err := os.Stat(ollamaPath); err != nil {
		pwd, err := os.Getwd()
		if err != nil {
			slog.Warn("missing ollama.exe and failed to get pwd", "error", err)
			return
		}
		distAppPath := filepath.Join(pwd, "dist", "windows-"+runtime.GOARCH)
		distOllamaPath := filepath.Join(distAppPath, "ollama.exe")
		if _, err := os.Stat(distOllamaPath); err == nil {
			slog.Info("detected developer mode")
			appPath = distAppPath
			ollamaPath = distOllamaPath
		}
	}
}

func maybeMoveAndRestart() appMove {
	return 0
}

// handleExistingInstance checks for existing instances and optionally focuses them
func handleExistingInstance(startHidden bool) bool {
	if wintray.CheckAndFocusExistingInstance(!startHidden) {
		slog.Info("existing instance found, exiting")
		os.Exit(0)
	}
	return true
}

func installSymlink() {}

type appCallbacks struct {
	t        wintray.TrayCallbacks
	shutdown func()
}

var app = &appCallbacks{}

// ShowSettings is called when people open Ollama while it is running.
func (*appCallbacks) ShowSettings() {
	showSettings(settingsPaneDefault)
}

// showSettings opens the settings window at pane.
func showSettings(pane settingsPane) {
	// TODO: add the native Windows settings window.
	slog.Info("settings window is not available on Windows yet", "pane", pane)
}

func (app *appCallbacks) Quit() {
	app.t.Quit()
}

func quit() {
	if app.t != nil {
		app.t.Quit()
	}
}

func (app *appCallbacks) DoUpdate() {
	// Safeguard in case we have requests in flight that need to drain...
	slog.Info("Waiting for server to shutdown")

	app.shutdown()

	if err := updater.DoUpgrade(true); err != nil { //nolint:staticcheck,nolintlint // DoUpgrade may always return non-nil on Windows
		slog.Warn(fmt.Sprintf("upgrade attempt failed: %s", err))
	}
}

// HandleURLScheme implements the URLSchemeHandler interface
func (app *appCallbacks) HandleURLScheme(urlScheme string) {
	handleURLSchemeRequest(urlScheme)
}

// handleURLSchemeRequest processes URL scheme requests from other instances
func handleURLSchemeRequest(urlScheme string) {
	handleURLSchemeInCurrentInstance(urlScheme)
}

func UpdateAvailable(ver string) error {
	if app.t == nil {
		slog.Debug("tray not yet initialized, skipping update notification")
		return nil
	}
	return app.t.UpdateAvailable(ver)
}

func osRun(shutdown func(), hasCompletedFirstRun, startHidden bool, urlSchemeRequest string) {
	var err error
	app.shutdown = shutdown
	app.t, err = wintray.NewTray(app)
	if err != nil {
		log.Fatalf("Failed to start: %s", err)
	}

	// Check for pending updates now that the tray is initialized.
	// The platform-independent check in app.go fires before osRun,
	// when app.t is still nil, so we must re-check here.
	if updater.IsUpdatePending() {
		slog.Debug("update pending on startup, showing tray notification")
		UpdateAvailable("")
	}

	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)

	// TODO - can this be generalized?
	go func() {
		<-signals
		slog.Debug("shutting down due to signal")
		app.t.Quit()
	}()

	// On windows, we run the final tasks in the main thread
	// before starting the tray event loop.  These final tasks
	// may trigger the UI, and must do that from the main thread.
	if !startHidden {
		// Determine if the process was started from a shortcut
		// ~\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\Ollama
		const STARTF_TITLEISLINKNAME = 0x00000800
		var info windows.StartupInfo
		if err := windows.GetStartupInfo(&info); err != nil {
			slog.Debug("unable to retrieve startup info", "error", err)
		} else if info.Flags&STARTF_TITLEISLINKNAME == STARTF_TITLEISLINKNAME {
			linkPath := windows.UTF16PtrToString(info.Title)
			if strings.Contains(linkPath, "Startup") {
				startHidden = true
			}
		}
	}
	runInitialUI(startHidden, urlSchemeRequest, startHiddenTasks, handleURLSchemeInCurrentInstance, showSettings)

	if !hasCompletedFirstRun {
		// Only create the login shortcut on first start
		// so we can respect users deletion of the link
		err = createLoginShortcut()
		if err != nil {
			slog.Warn("unable to create login shortcut", "error", err)
		}
	}

	app.t.TrayRun() // This will block the main thread
}

func createLoginShortcut() error {
	// The installer lays down a shortcut for us so we can copy it without
	// having to resort to calling COM APIs to establish the shortcut
	shortcutOrigin := filepath.Join(appPath, "lib", "Ollama.lnk")

	_, err := os.Stat(startupShortcut)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			in, err := os.Open(shortcutOrigin)
			if err != nil {
				return fmt.Errorf("unable to open shortcut %s : %w", shortcutOrigin, err)
			}
			defer in.Close()
			out, err := os.Create(startupShortcut)
			if err != nil {
				return fmt.Errorf("unable to open startup link %s : %w", startupShortcut, err)
			}
			defer out.Close()
			_, err = io.Copy(out, in)
			if err != nil {
				return fmt.Errorf("unable to copy shortcut %s : %w", startupShortcut, err)
			}
			err = out.Sync()
			if err != nil {
				return fmt.Errorf("unable to sync shortcut %s : %w", startupShortcut, err)
			}
			slog.Info("Created Startup shortcut", "shortcut", startupShortcut)
		} else {
			slog.Warn("unexpected error looking up Startup shortcut", "error", err)
		}
	} else {
		slog.Debug("Startup link already exists", "shortcut", startupShortcut)
	}
	return nil
}

func LaunchNewApp() {
}

func logStartup() {
	slog.Info("starting Ollama", "app", appPath, "version", version.Version, "OS", updater.UserAgentOS)
}

func runInBackground() {
	exe, err := os.Executable()
	if err != nil {
		slog.Error("failed to get executable path", "error", err)
		os.Exit(1)
	}
	cmd := exec.Command(exe, "hidden")
	if cmd != nil {
		err = cmd.Run()
		if err != nil {
			slog.Error("failed to run Ollama", "exe", exe, "error", err)
			os.Exit(1)
		}
	} else {
		slog.Error("failed to start Ollama", "exe", exe)
		os.Exit(1)
	}
}

// checkAndHandleExistingInstance checks if another instance is running and sends the URL to it
func checkAndHandleExistingInstance(urlSchemeRequest string) {
	if urlSchemeRequest == "" {
		return
	}

	// Try to send URL to existing instance using wintray messaging
	if wintray.CheckAndSendToExistingInstance(urlSchemeRequest) {
		os.Exit(0)
	}
}
