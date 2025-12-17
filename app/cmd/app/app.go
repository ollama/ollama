//go:build windows || darwin

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/ollama/ollama/app/auth"
	"github.com/ollama/ollama/app/logrotate"
	"github.com/ollama/ollama/app/server"
	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/tools"
	"github.com/ollama/ollama/app/ui"
	"github.com/ollama/ollama/app/updater"
	"github.com/ollama/ollama/app/version"
)

var (
	wv           = &Webview{}
	uiServerPort int
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
				// Development mode: use local dev server and enable CORS
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
		if checkAndHandleExistingInstance(urlSchemeRequest) {
			// The function will exit if it successfully sends to another instance
			// If we reach here, we're the first/only instance
		} else {
			// No existing instance found, handle the URL scheme in this instance
			go func() {
				handleURLSchemeInCurrentInstance(urlSchemeRequest)
			}()
		}
	}

	if u := os.Getenv("OLLAMA_UPDATE_URL"); u != "" {
		updater.UpdateCheckURLBase = u
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
	handleExistingInstance(startHidden)

	// on macOS, offer the user to create a symlink
	// from /usr/local/bin/ollama to the app bundle
	installSymlink()

	var ln net.Listener
	if devMode {
		// Use a fixed port in dev mode for predictable API access
		ln, err = net.Listen("tcp", "127.0.0.1:3001")
	} else {
		ln, err = net.Listen("tcp", "127.0.0.1:0")
	}
	if err != nil {
		slog.Error("failed to find available port", "error", err)
		return
	}

	port := ln.Addr().(*net.TCPAddr).Port
	token := uuid.NewString()
	wv.port = port
	wv.token = token
	uiServerPort = port

	st := &store.Store{}

	// Enable CORS in development mode
	if devMode {
		os.Setenv("OLLAMA_CORS", "1")

		// Check if Vite dev server is running on port 5173
		var conn net.Conn
		var err error
		for _, addr := range []string{"127.0.0.1:5173", "localhost:5173"} {
			conn, err = net.DialTimeout("tcp", addr, 2*time.Second)
			if err == nil {
				conn.Close()
				break
			}
		}

		if err != nil {
			slog.Error("Vite dev server not running on port 5173")
			fmt.Fprintln(os.Stderr, "Error: Vite dev server is not running on port 5173")
			fmt.Fprintln(os.Stderr, "Please run 'npm run dev' in the ui/app directory to start the UI in development mode")
			os.Exit(1)
		}
	}

	// Initialize tools registry
	toolRegistry := tools.NewRegistry()
	slog.Info("initialized tools registry", "tool_count", len(toolRegistry.List()))

	// ctx is the app-level context that will be used to stop the app
	ctx, cancel := context.WithCancel(context.Background())

	// octx is the ollama server context that will be used to stop the ollama server
	octx, ocancel := context.WithCancel(ctx)

	// TODO (jmorganca): instead we should instantiate the
	// webview with the store instead of assigning it here, however
	// making the webview a global variable is easier for now
	wv.Store = st
	done := make(chan error, 1)
	osrv := server.New(st, devMode)
	go func() {
		slog.Info("starting ollama server")
		done <- osrv.Run(octx)
	}()

	upd := &updater.Updater{Store: st}

	uiServer := ui.Server{
		Token: token,
		Restart: func() {
			ocancel()
			<-done
			octx, ocancel = context.WithCancel(ctx)
			go func() {
				done <- osrv.Run(octx)
			}()
		},
		Store:        st,
		ToolRegistry: toolRegistry,
		Dev:          devMode,
		Logger:       slog.Default(),
		Updater:      upd,
		UpdateAvailableFunc: func() {
			UpdateAvailable("")
		},
		ClearUpdateAvailableFunc: func() {
			ClearUpdateAvailable()
		},
	}

	srv := &http.Server{
		Handler: uiServer.Handler(),
	}

	// Start the UI server
	slog.Info("starting ui server", "port", port)
	go func() {
		slog.Debug("starting ui server on port", "port", port)
		err = srv.Serve(ln)
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Warn("desktop server", "error", err)
		}
		slog.Debug("background desktop server done")
	}()

	upd.StartBackgroundUpdaterChecker(ctx, UpdateAvailable)

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

	if urlSchemeRequest != "" {
		go func() {
			handleURLSchemeInCurrentInstance(urlSchemeRequest)
		}()
	} else {
		slog.Debug("no URL scheme request to handle")
	}

	go func() {
		slog.Debug("waiting for ollama server to be ready")
		if err := ui.WaitForServer(ctx, 10*time.Second); err != nil {
			slog.Warn("ollama server not ready, continuing anyway", "error", err)
		}

		if _, err := uiServer.UserData(ctx); err != nil {
			slog.Warn("failed to load user data", "error", err)
		}
	}()

	osRun(cancel, hasCompletedFirstRun, startHidden)

	slog.Info("shutting down desktop server")
	if err := srv.Close(); err != nil {
		slog.Warn("error shutting down desktop server", "error", err)
	}

	slog.Info("shutting down ollama server")
	cancel()
	<-done
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
			// Check if auto-update is enabled before upgrading
			st := &store.Store{}
			settings, err := st.Settings()
			if err != nil {
				slog.Warn("failed to load settings for upgrade check", "error", err)
			} else if !settings.AutoUpdateEnabled {
				slog.Info("auto-update disabled, skipping automatic upgrade at startup")
				return
			}

			if err := updater.DoUpgradeAtStartup(); err != nil {
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

func checkUserLoggedIn(uiServerPort int) bool {
	if uiServerPort == 0 {
		slog.Debug("UI server not ready yet, skipping auth check")
		return false
	}

	resp, err := http.Post(fmt.Sprintf("http://127.0.0.1:%d/api/me", uiServerPort), "application/json", nil)
	if err != nil {
		slog.Debug("failed to call local auth endpoint", "error", err)
		return false
	}
	defer resp.Body.Close()

	// Check if the response is successful
	if resp.StatusCode != http.StatusOK {
		slog.Debug("auth endpoint returned non-OK status", "status", resp.StatusCode)
		return false
	}

	var user struct {
		ID   string `json:"id"`
		Name string `json:"name"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&user); err != nil {
		slog.Debug("failed to parse user response", "error", err)
		return false
	}

	// Verify we have a valid user with an ID and name
	if user.ID == "" || user.Name == "" {
		slog.Debug("user response missing required fields", "id", user.ID, "name", user.Name)
		return false
	}

	slog.Debug("user is logged in", "user_id", user.ID, "user_name", user.Name)
	return true
}

// handleConnectURLScheme fetches the connect URL and opens it in the browser
func handleConnectURLScheme() {
	if checkUserLoggedIn(uiServerPort) {
		slog.Info("user is already logged in, opening app instead")
		showWindow(wv.webview.Window())
		return
	}

	connectURL, err := auth.BuildConnectURL("https://ollama.com")
	if err != nil {
		slog.Error("failed to build connect URL", "error", err)
		openInBrowser("https://ollama.com/connect")
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
// Supports: ollama:// (open app) and ollama://connect (OAuth)
func parseURLScheme(urlSchemeRequest string) (isConnect bool, err error) {
	parsedURL, err := url.Parse(urlSchemeRequest)
	if err != nil {
		return false, fmt.Errorf("invalid URL: %w", err)
	}

	// Check if this is a connect URL
	if parsedURL.Host == "connect" || strings.TrimPrefix(parsedURL.Path, "/") == "connect" {
		return true, nil
	}

	// Allow bare ollama:// or ollama:/// to open the app
	if (parsedURL.Host == "" && parsedURL.Path == "") || parsedURL.Path == "/" {
		return false, nil
	}

	return false, fmt.Errorf("unsupported ollama:// URL path: %s", urlSchemeRequest)
}

// handleURLSchemeInCurrentInstance processes URL scheme requests in the current instance
func handleURLSchemeInCurrentInstance(urlSchemeRequest string) {
	isConnect, err := parseURLScheme(urlSchemeRequest)
	if err != nil {
		slog.Error("failed to parse URL scheme request", "url", urlSchemeRequest, "error", err)
		return
	}

	if isConnect {
		handleConnectURLScheme()
	} else {
		if wv.webview != nil {
			showWindow(wv.webview.Window())
		}
	}
}
