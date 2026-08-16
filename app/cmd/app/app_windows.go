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
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/ollama/ollama/app/dialog"
	"github.com/ollama/ollama/app/updater"
	"github.com/ollama/ollama/app/version"
	"github.com/ollama/ollama/app/wintray"
	"golang.org/x/sys/windows"
)

const windowsAppRuntimeDownloadURL = "https://learn.microsoft.com/windows/apps/windows-app-sdk/downloads-archive#windows-app-sdk-18"

var (
	u32                  = windows.NewLazySystemDLL("User32.dll")
	pBringWindowToTop    = u32.NewProc("BringWindowToTop")
	pShowWindow          = u32.NewProc("ShowWindow")
	pSendMessage         = u32.NewProc("SendMessageA")
	pGetSystemMetrics    = u32.NewProc("GetSystemMetrics")
	pGetWindowRect       = u32.NewProc("GetWindowRect")
	pSetWindowPos        = u32.NewProc("SetWindowPos")
	pSetForegroundWindow = u32.NewProc("SetForegroundWindow")
	pSetActiveWindow     = u32.NewProc("SetActiveWindow")
	pIsIconic            = u32.NewProc("IsIconic")
	pSetWindowLongPtr    = u32.NewProc("SetWindowLongPtrW")
	pCallWindowProc      = u32.NewProc("CallWindowProcW")
	pDefWindowProc       = u32.NewProc("DefWindowProcW")

	// Publish originalProc before handle and clear handle before originalProc.
	// A matching nonzero handle therefore guarantees that the callback can
	// forward messages through the corresponding original procedure.
	mainWindowHandle       atomic.Uintptr
	mainWindowOriginalProc atomic.Uintptr
	mainWindowCloseProc    = windows.NewCallback(mainWindowProc)
	mainWindowProcMu       sync.Mutex // serializes the compound native subclass operation

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
func handleExistingInstance(startHidden bool) {
	if wintray.CheckAndFocusExistingInstance(!startHidden) {
		slog.Info("existing instance found, exiting")
		os.Exit(0)
	}

	// Fail before the server supervisor is constructed so a UI initialization
	// failure cannot disturb managed server state.
	if err := wintray.Initialize(); err != nil {
		slog.Error("Windows UI is unavailable", "error", err)
		if errors.Is(err, wintray.ErrWindowsAppRuntimeUnavailable) {
			showMissingWindowsAppRuntimeDialog()
		}
		os.Exit(1)
	}
}

func showMissingWindowsAppRuntimeDialog() {
	dialog.Message(
		"Ollama requires Windows App Runtime 1.8, but it is missing or damaged.\n\n"+
			"Download it from Microsoft:\n%s\n\n"+
			"After installing it, start Ollama again. "+
			"If you are running Ollama from a ZIP, make sure you extracted the complete archive. "+
			"If you installed Ollama with OllamaSetup.exe, reinstalling Ollama will also restore this dependency.",
		windowsAppRuntimeDownloadURL,
	).Title("Ollama couldn't start").Error()
}

func installSymlink() {}

type appCallbacks struct {
	t        wintray.TrayCallbacks
	shutdown func()
}

var (
	app         = &appCallbacks{}
	appQuitOnce sync.Once
)

func (ac *appCallbacks) UIRun(path string) {
	runUI(path)
}

func (*appCallbacks) UIShow() {
	if !wv.show() {
		runUI("/")
	}
}

func (*appCallbacks) UITerminate() {
	terminateUI()
}

func (*appCallbacks) UIRunning() bool {
	return wv.IsRunning()
}

func (*appCallbacks) ShowLogs() error {
	logDir := filepath.Dir(appLogPath)
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		return fmt.Errorf("create log directory: %w", err)
	}
	verb, err := windows.UTF16PtrFromString("open")
	if err != nil {
		return fmt.Errorf("encode shell action: %w", err)
	}
	path, err := windows.UTF16PtrFromString(logDir)
	if err != nil {
		return fmt.Errorf("encode log directory: %w", err)
	}

	slog.Debug("opening Ollama log directory", "path", logDir)
	if err := windows.ShellExecute(0, verb, path, nil, nil, windows.SW_SHOWNORMAL); err != nil {
		return fmt.Errorf("open log directory: %w", err)
	}
	return nil
}

func (app *appCallbacks) Quit() {
	quit()
}

// TODO - reconcile with above for consistency between mac/windows
func quit() {
	appQuitOnce.Do(func() {
		// Stop background work before releasing tray and WebView2 resources. This
		// prevents a late updater callback from posting to a destroyed IUP handle.
		if app.shutdown != nil {
			app.shutdown()
		}

		// Queue WebView2 destruction before stopping the shared IUP/WinUI message
		// loop so its thread-affine COM objects are released on the UI thread.
		terminateUI()
		if app.t != nil {
			app.t.Quit()
		}
	})
}

func (app *appCallbacks) DoUpdate() {
	// Safeguard in case we have requests in flight that need to drain...
	slog.Info("Waiting for server to shutdown")

	app.shutdown()

	if err := updater.DoUpgrade(true); err != nil {
		slog.Warn(fmt.Sprintf("upgrade attempt failed: %s", err))
	}
}

// HandleURLScheme implements the URLSchemeHandler interface
func (app *appCallbacks) HandleURLScheme(urlScheme string) {
	handleURLSchemeRequest(urlScheme)
}

// handleURLSchemeRequest processes URL scheme requests from other instances
func handleURLSchemeRequest(urlScheme string) {
	isConnect, err := parseURLScheme(urlScheme)
	if err != nil {
		slog.Error("failed to parse URL scheme request", "url", urlScheme, "error", err)
		return
	}

	if isConnect {
		handleConnectURLScheme()
	} else {
		wv.show()
	}
}

func UpdateAvailable(ver string) error {
	if app.t == nil {
		slog.Debug("tray not yet initialized, skipping update notification")
		return nil
	}
	return app.t.UpdateAvailable(ver)
}

func osRun(shutdown func(), hasCompletedFirstRun, startHidden bool) {
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
	if startHidden {
		startHiddenTasks()
	} else {
		ptr := runUI("/")

		// Set the window icon using the tray icon
		if ptr != nil {
			iconHandle := app.t.GetIconHandle()
			if iconHandle != 0 {
				hwnd := uintptr(ptr)
				const ICON_SMALL = 0
				const ICON_BIG = 1
				const WM_SETICON = 0x0080

				pSendMessage.Call(hwnd, uintptr(WM_SETICON), uintptr(ICON_SMALL), uintptr(iconHandle))
				pSendMessage.Call(hwnd, uintptr(WM_SETICON), uintptr(ICON_BIG), uintptr(iconHandle))
			}
		}

		centerWindow(ptr)
	}

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

const (
	SW_HIDE        = 0  // Hides the window
	SW_SHOW        = 5  // Shows window in its current size/position
	SW_SHOWNA      = 8  // Shows without activating
	SW_MINIMIZE    = 6  // Minimizes the window
	SW_RESTORE     = 9  // Restores to previous size/position
	SW_SHOWDEFAULT = 10 // Sets show state based on program state
	SM_CXSCREEN    = 0
	SM_CYSCREEN    = 1
	HWND_TOP       = 0
	SWP_NOSIZE     = 0x0001
	SWP_NOMOVE     = 0x0002
	SWP_NOZORDER   = 0x0004
	SWP_SHOWWINDOW = 0x0040

	// Menu constants
	MF_STRING     = 0x00000000
	MF_SEPARATOR  = 0x00000800
	MF_GRAYED     = 0x00000001
	TPM_RETURNCMD = 0x0100
)

// POINT structure for cursor position
type POINT struct {
	X int32
	Y int32
}

// Rect structure for GetWindowRect
type Rect struct {
	Left   int32
	Top    int32
	Right  int32
	Bottom int32
}

func centerWindow(ptr unsafe.Pointer) {
	hwnd := uintptr(ptr)
	if hwnd == 0 {
		return
	}

	var rect Rect
	pGetWindowRect.Call(hwnd, uintptr(unsafe.Pointer(&rect)))

	screenWidth, _, _ := pGetSystemMetrics.Call(uintptr(SM_CXSCREEN))
	screenHeight, _, _ := pGetSystemMetrics.Call(uintptr(SM_CYSCREEN))

	windowWidth := rect.Right - rect.Left
	windowHeight := rect.Bottom - rect.Top

	x := (int32(screenWidth) - windowWidth) / 2
	y := (int32(screenHeight) - windowHeight) / 2

	// Ensure the window is not positioned off-screen
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}

	pSetWindowPos.Call(
		hwnd,
		uintptr(HWND_TOP),
		uintptr(x),
		uintptr(y),
		uintptr(windowWidth),  // Keep original width
		uintptr(windowHeight), // Keep original height
		uintptr(SWP_SHOWWINDOW),
	)
}

func showWindow(ptr unsafe.Pointer) {
	hwnd := uintptr(ptr)
	if hwnd != 0 {
		iconHandle := app.t.GetIconHandle()
		if iconHandle != 0 {
			const ICON_SMALL = 0
			const ICON_BIG = 1
			const WM_SETICON = 0x0080

			pSendMessage.Call(hwnd, uintptr(WM_SETICON), uintptr(ICON_SMALL), uintptr(iconHandle))
			pSendMessage.Call(hwnd, uintptr(WM_SETICON), uintptr(ICON_BIG), uintptr(iconHandle))
		}

		// Check if window is minimized
		isMinimized, _, _ := pIsIconic.Call(hwnd)
		if isMinimized != 0 {
			// Restore the window if it's minimized
			pShowWindow.Call(hwnd, uintptr(SW_RESTORE))
		}

		// Show the window
		pShowWindow.Call(hwnd, uintptr(SW_SHOW))

		// Bring window to top
		pBringWindowToTop.Call(hwnd)

		// Force window to foreground
		pSetForegroundWindow.Call(hwnd)

		// Make it the active window
		pSetActiveWindow.Call(hwnd)

		// Ensure window is positioned on top
		pSetWindowPos.Call(
			hwnd,
			uintptr(HWND_TOP),
			0, 0, 0, 0,
			uintptr(SWP_NOSIZE|SWP_NOMOVE|SWP_SHOWWINDOW),
		)
	}
}

// HideWindow hides the application window
func hideWindow(ptr unsafe.Pointer) {
	hwnd := uintptr(ptr)
	if hwnd != 0 {
		pShowWindow.Call(
			hwnd,
			uintptr(SW_HIDE),
		)
	}
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

func drag(ptr unsafe.Pointer) {}

func doubleClick(ptr unsafe.Pointer) {}

// checkAndHandleExistingInstance checks if another instance is running and sends the URL to it
func checkAndHandleExistingInstance(urlSchemeRequest string) bool {
	if urlSchemeRequest == "" {
		return false
	}
	// Try to send URL to existing instance using wintray messaging
	if wintray.CheckAndSendToExistingInstance(urlSchemeRequest) {
		os.Exit(0)
		return true
	}

	// No existing instance, we'll handle it ourselves
	return false
}

const (
	gwlpWndProc = ^uintptr(3) // GWLP_WNDPROC (-4)
	wmClose     = 0x0010
)

func runUI(path string) unsafe.Pointer {
	window := wv.Run(path)
	if err := makeMainWindowCloseToTray(window); err != nil {
		slog.Error("failed to make main window close to tray", "error", err)
	}
	return window
}

func makeMainWindowCloseToTray(window unsafe.Pointer) error {
	mainWindowProcMu.Lock()
	defer mainWindowProcMu.Unlock()

	hwnd := uintptr(window)
	if hwnd == 0 || mainWindowHandle.Load() == hwnd {
		return nil
	}
	if err := restoreMainWindowProcLocked(); err != nil {
		return err
	}

	original, _, callErr := pSetWindowLongPtr.Call(hwnd, gwlpWndProc, mainWindowCloseProc)
	if original == 0 {
		return fmt.Errorf("subclass main window: %w", callErr)
	}
	mainWindowOriginalProc.Store(original)
	mainWindowHandle.Store(hwnd)
	return nil
}

func mainWindowProc(hwnd windows.Handle, message uint32, wParam, lParam uintptr) uintptr {
	if message == wmClose {
		slog.Debug("hiding main window after native close")
		pShowWindow.Call(uintptr(hwnd), SW_HIDE) //nolint:errcheck
		return 0
	}

	original := mainWindowOriginalProc.Load()
	if mainWindowHandle.Load() == uintptr(hwnd) && original != 0 {
		result, _, _ := pCallWindowProc.Call(original, uintptr(hwnd), uintptr(message), wParam, lParam)
		return result
	}
	result, _, _ := pDefWindowProc.Call(uintptr(hwnd), uintptr(message), wParam, lParam)
	return result
}

func restoreMainWindowProc() error {
	mainWindowProcMu.Lock()
	defer mainWindowProcMu.Unlock()
	return restoreMainWindowProcLocked()
}

func restoreMainWindowProcLocked() error {
	hwnd := mainWindowHandle.Load()
	original := mainWindowOriginalProc.Load()
	if hwnd == 0 || original == 0 {
		return nil
	}

	// Stop intercepting window messages before WebView2 tears down the native
	// window. In particular, its destruction messages must reach WebView2's
	// original procedure rather than DefWindowProc.
	previous, _, callErr := pSetWindowLongPtr.Call(hwnd, gwlpWndProc, original)
	if previous == 0 {
		return fmt.Errorf("restore main window procedure: %w", callErr)
	}
	mainWindowHandle.CompareAndSwap(hwnd, 0)
	mainWindowOriginalProc.CompareAndSwap(original, 0)
	return nil
}

func terminateUI() {
	if err := restoreMainWindowProc(); err != nil {
		slog.Warn("failed to restore main window procedure", "error", err)
	}
	wv.Terminate()
}

func (w *Webview) show() bool {
	w.mutex.Lock()
	defer w.mutex.Unlock()
	if w.webview == nil {
		return false
	}
	showWindow(w.webview.Window())
	return true
}
