//go:build windows || darwin

package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"

	"github.com/ollama/ollama/app/logrotate"
	"github.com/ollama/ollama/app/server"
	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/updater"
	"github.com/ollama/ollama/app/version"
)

var (
	appStore *store.Store
	settings *settingsController
)

var debug = strings.EqualFold(os.Getenv("OLLAMA_DEBUG"), "true") || os.Getenv("OLLAMA_DEBUG") == "1"

var (
	fastStartup = false
	devMode     = false
)

type appMove int

const (
	CannotMove appMove = iota
	UserDeclinedMove
	MoveCompleted
	AlreadyMoved
	LoginSession
	PermissionDenied
	MoveError
)

func main() {
	startHidden := false
	var urlSchemeRequest string
	if len(os.Args) > 1 {
		for _, arg := range os.Args {
			// Handle URL scheme requests (Windows)
			if strings.HasPrefix(arg, "ollama://") {
				urlSchemeRequest = arg
				slog.Info("received URL scheme request", "url", arg)
				continue
			}
			switch arg {
			case "serve":
				fmt.Fprintln(os.Stderr, "serve command not supported, use ollama")
				os.Exit(1)
			case "version", "-v", "--version":
				fmt.Println(version.Version)
				os.Exit(0)
			case "background":
				// When running the process in this "background" mode, we spawn a
				// child process for the main app.  This is necessary so the
				// "Allow in the Background" setting in MacOS can be unchecked
				// without breaking the main app.  Two copies of the app are
				// present in the bundle, one for the main app and one for the
				// background initiator.
				fmt.Fprintln(os.Stdout, "starting in background")
				runInBackground()
				os.Exit(0)
			case "hidden", "-j", "--hide":
				// startHidden suppresses the UI on startup, and can be triggered multiple ways
				// On windows, path based via login startup detection
				// On MacOS via [NSApp isHidden] from `open -j -a /Applications/Ollama.app` or equivalent
				// On both via the "hidden" command line argument
				startHidden = true
			case "--fast-startup":
				// Skip optional steps like pending updates to start quickly for immediate use
				fastStartup = true
			case "-dev", "--dev":
				// Development mode: never stop other ollama servers, and allow
				// OLLAMA_APP_DB_PATH to isolate the app database
				devMode = true
			}
		}
	}

	level := slog.LevelInfo
	if debug {
		level = slog.LevelDebug
	}

	logrotate.Rotate(appLogPath)
	if _, err := os.Stat(filepath.Dir(appLogPath)); errors.Is(err, os.ErrNotExist) {
		if err := os.MkdirAll(filepath.Dir(appLogPath), 0o755); err != nil {
			slog.Error(fmt.Sprintf("failed to create server log dir %v", err))
			return
		}
	}

	var logFile io.Writer
	var err error
	logFile, err = os.OpenFile(appLogPath, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0o755)
	if err != nil {
		slog.Error(fmt.Sprintf("failed to create server log %v", err))
		return
	}
	// Detect if we're a GUI app on windows, and if not, send logs to console as well
	if os.Stderr.Fd() != 0 {
		// Console app detected
		logFile = io.MultiWriter(os.Stderr, logFile)
	}

	handler := slog.NewTextHandler(logFile, &slog.HandlerOptions{
		Level:     level,
		AddSource: true,
		ReplaceAttr: func(_ []string, attr slog.Attr) slog.Attr {
			if attr.Key == slog.SourceKey {
				source := attr.Value.Any().(*slog.Source)
				source.File = filepath.Base(source.File)
			}
			return attr
		},
	})

	slog.SetDefault(slog.New(handler))
	logStartup()

	// On Windows, check if another instance is running and send URL to it
	// Do this after logging is set up so we can debug issues
	if runtime.GOOS == "windows" && urlSchemeRequest != "" {
		slog.Debug("checking for existing instance", "url", urlSchemeRequest)
		// This exits after forwarding the request when another instance is
		// running. First-instance requests are handled later by osRun, after the
		// Windows UI dependencies are initialized and from the primary thread.
		checkAndHandleExistingInstance(urlSchemeRequest)
	}

	// Detect if this is a first start after an upgrade, in
	// which case we need to do some cleanup
	var skipMove bool
	if _, err := os.Stat(updater.UpgradeMarkerFile); err == nil {
		slog.Debug("first start after upgrade")
		err = updater.DoPostUpgradeCleanup()
		if err != nil {
			slog.Error("failed to cleanup prior version", "error", err)
		}
		// We never prompt to move the app after an upgrade
		skipMove = true
		// Start hidden after updates to prevent UI from opening automatically
		startHidden = true
	}

	if !skipMove && !fastStartup {
		if maybeMoveAndRestart() == MoveCompleted {
			return
		}
	}

	// Check if another instance is already running
	// On Windows, focus the existing instance; on other platforms, kill it
	if !handleExistingInstance(startHidden) {
		return
	}

	// on macOS, offer the user to create a symlink
	// from /usr/local/bin/ollama to the app bundle
	installSymlink()

	st := &store.Store{}
	if devMode {
		if dbPath := strings.TrimSpace(os.Getenv("OLLAMA_APP_DB_PATH")); dbPath != "" {
			st.DBPath = dbPath
			slog.Debug("using development app database", "path", dbPath)
		}
	}
	appStore = st

	// ctx is the app-level context that will be used to stop the app
	ctx, cancel := context.WithCancel(context.Background())

	ollama := &ollamaServer{server: server.New(st, devMode)}
	ollama.Start(ctx)

	upd := &updater.Updater{Store: st}
	settings = &settingsController{
		store:         st,
		restartServer: ollama.Restart,
		updater:       upd,
		notifyUpdate:  func() { UpdateAvailable("") },
	}
	upd.StartBackgroundUpdaterChecker(ctx, UpdateAvailable)

	// Check for pending updates on startup (show tray notification if update is ready)
	if updater.IsUpdatePending() {
		// On Windows, the tray is initialized in osRun(). Calling UpdateAvailable
		// before that would dereference a nil tray callback.
		// TODO: refactor so the update check runs after platform init on all platforms.
		if runtime.GOOS == "windows" {
			slog.Debug("update pending on startup, deferring tray notification until tray initialization")
		} else {
			slog.Debug("update pending on startup, showing tray notification")
			UpdateAvailable("")
		}
	}

	hasCompletedFirstRun, err := st.HasCompletedFirstRun()
	if err != nil {
		slog.Error("failed to load has completed first run", "error", err)
	}

	if !hasCompletedFirstRun {
		err = st.SetHasCompletedFirstRun(true)
		if err != nil {
			slog.Error("failed to set has completed first run", "error", err)
		}
	}

	// capture SIGINT and SIGTERM signals and gracefully shutdown the app
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-signals
		slog.Info("received SIGINT or SIGTERM signal, shutting down")
		quit()
	}()

	if urlSchemeRequest != "" && runtime.GOOS != "windows" {
		go func() {
			handleURLSchemeInCurrentInstance(urlSchemeRequest)
		}()
	} else if urlSchemeRequest == "" {
		slog.Debug("no URL scheme request to handle")
	}

	// Refresh the cached account so Settings can show it without waiting.
	go func() {
		if _, err := settings.Account(ctx); err != nil {
			slog.Debug("failed to refresh account", "error", err)
		}
	}()

	osRun(cancel, hasCompletedFirstRun, startHidden, urlSchemeRequest)

	slog.Info("shutting down ollama server")
	cancel()
	ollama.Wait()
}

// runInitialUI decides what to show once the platform UI is ready: a URL
// scheme request wins, a hidden launch shows nothing, and an interactive
// launch opens Settings so people can see that Ollama is running.
func runInitialUI(startHidden bool, urlSchemeRequest string, startHiddenFn func(), handleURLFn func(string), showSettingsFn func(settingsPane)) {
	switch {
	case urlSchemeRequest != "":
		handleURLFn(urlSchemeRequest)
	case startHidden:
		startHiddenFn()
	default:
		showSettingsFn(settingsPaneDefault)
	}
}

func startHiddenTasks() {
	// If an upgrade is ready and we're in hidden mode, perform it at startup.
	// If we're not in hidden mode, we want to start as fast as possible and not
	// slow the user down with an upgrade.
	if updater.IsUpdatePending() {
		if fastStartup {
			// CLI triggered app startup use-case
			slog.Info("deferring pending update for fast startup")
		} else {
			// Check if auto-update is enabled before automatically upgrading
			settings, err := appStore.Settings()
			if err != nil {
				slog.Warn("failed to load settings for upgrade check", "error", err)
			} else if !settings.AutoUpdateEnabled {
				slog.Info("auto-update disabled, skipping automatic upgrade at startup")
				// Still show tray notification so user knows update is ready
				UpdateAvailable("")
				return
			}

			if err := updater.DoUpgradeAtStartup(); err != nil { //nolint:staticcheck,nolintlint // DoUpgradeAtStartup may always return non-nil on Windows
				slog.Info("unable to perform upgrade at startup", "error", err)
				// Make sure the restart to upgrade menu shows so we can attempt an interactive upgrade to get authorization
				UpdateAvailable("")
			} else {
				slog.Debug("launching new version...")
				// TODO - consider a timer that aborts if this takes too long and we haven't been killed yet...
				LaunchNewApp()
				os.Exit(0)
			}
		}
	}
}

// handleConnectURLScheme starts signing in to ollama.com in the browser, or
// shows the signed-in account when there is nothing to connect.
func handleConnectURLScheme() {
	if account, err := settings.Account(context.Background()); err == nil && account.SignedIn {
		slog.Info("user is already signed in, opening settings instead")
		showSettings(settingsPaneAccount)
		return
	}

	connectURL, err := settings.SignInURL()
	if err != nil {
		slog.Error("failed to build connect URL", "error", err)
		openInBrowser(ollamaDotCom + "/connect")
		return
	}

	openInBrowser(connectURL)
}

// openInBrowser opens the specified URL in the default browser
func openInBrowser(url string) {
	var cmd string
	var args []string

	switch runtime.GOOS {
	case "windows":
		cmd = "rundll32"
		args = []string{"url.dll,FileProtocolHandler", url}
	case "darwin":
		cmd = "open"
		args = []string{url}
	default: // "linux", "freebsd", "openbsd", "netbsd"... should not reach here
		slog.Warn("unsupported OS for openInBrowser", "os", runtime.GOOS)
	}

	slog.Info("executing browser command", "cmd", cmd, "args", args)
	if err := exec.Command(cmd, args...).Start(); err != nil {
		slog.Error("failed to open URL in browser", "url", url, "cmd", cmd, "args", args, "error", err)
	}
}

// parseURLScheme parses an ollama:// URL and validates it
// Supports: ollama:// (open settings), ollama://apps, and ollama://connect (sign in).
func parseURLScheme(urlSchemeRequest string) (action string, err error) {
	parsedURL, err := url.Parse(urlSchemeRequest)
	if err != nil {
		return "", fmt.Errorf("invalid URL: %w", err)
	}

	// Check if this is a connect URL
	if parsedURL.Host == "connect" || strings.TrimPrefix(parsedURL.Path, "/") == "connect" {
		return "connect", nil
	}

	if parsedURL.Host == "apps" || strings.TrimPrefix(parsedURL.Path, "/") == "apps" {
		return "apps", nil
	}

	// Allow bare ollama:// or ollama:/// to open the app
	if (parsedURL.Host == "" && parsedURL.Path == "") || parsedURL.Path == "/" {
		return "", nil
	}

	return "", fmt.Errorf("unsupported ollama:// URL path: %s", urlSchemeRequest)
}

// handleURLSchemeInCurrentInstance processes URL scheme requests in the current instance
func handleURLSchemeInCurrentInstance(urlSchemeRequest string) {
	err := dispatchURLSchemeRequest(urlSchemeRequest, handleConnectURLScheme, func() {
		showSettings(settingsPaneDefault)
	}, func() {
		showSettings(settingsPaneApps)
	})
	if err != nil {
		slog.Error("failed to parse URL scheme request", "url", urlSchemeRequest, "error", err)
	}
}

func dispatchURLSchemeRequest(urlSchemeRequest string, connect, open, apps func()) error {
	action, err := parseURLScheme(urlSchemeRequest)
	if err != nil {
		return err
	}
	switch action {
	case "connect":
		connect()
	case "apps":
		apps()
	default:
		open()
	}
	return nil
}
