//go:build windows || darwin

package main

// #cgo CFLAGS: -x objective-c
// #cgo LDFLAGS: -framework Webkit -framework Cocoa -framework LocalAuthentication -framework ServiceManagement
// #include "app_darwin.h"
// #include "../../updater/updater_darwin.h"
// typedef const char cchar_t;
import "C"

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/app/updater"
	"github.com/ollama/ollama/app/version"
	"github.com/ollama/ollama/cmd/launch"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/proxy"
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

type claudeProxyFailure uint8

const (
	claudeProxyFailureNone claudeProxyFailure = iota
	claudeProxyFailurePortConflict
)

var (
	isApp           = updater.BundlePath != ""
	appLogPath      = filepath.Join(os.Getenv("HOME"), ".ollama", "logs", "app.log")
	launchAgentPath = filepath.Join(os.Getenv("HOME"), "Library", "LaunchAgents", "com.ollama.ollama.plist")
	claudeAppProxy  *proxy.ClaudeDesktop
	claudeProxyMu   sync.Mutex
	claudeProxyErr  error
	claudeProxyFail claudeProxyFailure
	claudeDesktop   = &launch.ClaudeDesktop{}

	claudeDesktopInstalled = launch.ClaudeDesktopInstalled
	claudeProxyListenAddr  = proxy.DefaultClaudeDesktopListenAddr
	claudeProxyRetryWait   = 750 * time.Millisecond
	claudeProxyRetryPoll   = 50 * time.Millisecond
	claudeShutdownTimeout  = 30 * time.Second
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
	openUI("/")
}

func openUI(path string) {
	if wv.IsRunning() && wv.webview != nil {
		showWindow(wv.webview.Window())
		return
	}
	p := C.CString(path)
	defer C.free(unsafe.Pointer(p))
	StartUI(p)
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
	status := appMove(C.askToMoveToApplications())
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

func osRun(_ func(), hasCompletedFirstRun, startHidden, showOnboarding bool, _ string) {
	handoffSignal := make(chan os.Signal, 1)
	handoffDone := make(chan struct{})
	signal.Notify(handoffSignal, syscall.SIGUSR1)
	defer signal.Stop(handoffSignal)
	defer close(handoffDone)
	go func() {
		select {
		case <-handoffSignal:
			slog.Info("received app handoff signal, shutting down")
			stopClaudeAppProxy()
			C.quit()
		case <-handoffDone:
		}
	}()

	registerLaunchAgent(hasCompletedFirstRun)
	if err := reconcileClaudeAppProxy(); err != nil {
		slog.Warn("failed to start Claude gateway", "error", err)
	}
	defer stopClaudeAppProxy()

	// Run the native macOS app
	// Note: this will block until the app is closed
	slog.Debug("starting native darwin event loop")
	C.run(C._Bool(showOnboarding), C._Bool(startHidden))
}

func reconcileClaudeAppProxy() error {
	if !claudeDesktopInstalled() || !claudeDesktop.UsesOllamaGateway() {
		stopClaudeAppProxy()
		return nil
	}
	return startClaudeAppProxy()
}

func startClaudeAppProxy() error {
	claudeProxyMu.Lock()
	defer claudeProxyMu.Unlock()
	if claudeAppProxy != nil {
		clearClaudeProxyFailure()
		return nil
	}
	if !claudeDesktopInstalled() {
		slog.Debug("Claude Desktop is not installed; skipping gateway")
		clearClaudeProxyFailure()
		return nil
	}
	ollamaURL := envconfig.ConnectableHost()
	gatewayPort, err := claudeGatewayPort()
	if err != nil {
		return setClaudeProxyFailure(fmt.Errorf("parse Claude gateway address: %w", err), claudeProxyFailureNone)
	}
	if ollamaURL.Port() == gatewayPort {
		return setClaudeProxyFailure(
			fmt.Errorf("OLLAMA_HOST cannot use port %s because it is reserved for Claude", gatewayPort),
			claudeProxyFailurePortConflict,
		)
	}
	ollamaClient := api.NewClient(ollamaURL, http.DefaultClient)
	gateway, err := proxy.NewClaudeDesktop(proxy.ClaudeDesktopConfig{
		ListenAddr:      claudeProxyListenAddr,
		OllamaURL:       ollamaURL.String(),
		Model:           "glm-5.2:cloud",
		Logger:          slog.Default(),
		OnCountsChanged: updateClaudeProxyMenu,
		CloudModelsAvailable: func(ctx context.Context) bool {
			return claudeCloudModelsAvailable(ctx, ollamaClient.CloudStatusExperimental, ollamaClient.Whoami)
		},
		ListLocalModels: func(ctx context.Context) ([]string, error) {
			return claudeLocalModels(ctx, ollamaClient.List)
		},
	})
	if err != nil {
		return setClaudeProxyFailure(err, claudeProxyFailureNone)
	}
	if err := startClaudeGateway(gateway); err != nil {
		_ = gateway.Close(context.Background())
		failure := claudeProxyFailureNone
		if errors.Is(err, syscall.EADDRINUSE) {
			failure = claudeProxyFailurePortConflict
		}
		return setClaudeProxyFailure(err, failure)
	}
	claudeAppProxy = gateway
	clearClaudeProxyFailure()
	return nil
}

func startClaudeGateway(gateway *proxy.ClaudeDesktop) error {
	deadline := time.Now().Add(claudeProxyRetryWait)
	for {
		err := gateway.Start()
		if err == nil || !errors.Is(err, syscall.EADDRINUSE) || time.Now().After(deadline) {
			return err
		}
		time.Sleep(claudeProxyRetryPoll)
	}
}

func claudeCloudModelsAvailable(
	ctx context.Context,
	cloudStatus func(context.Context) (*api.StatusResponse, error),
	whoami func(context.Context) (*api.UserResponse, error),
) bool {
	ctx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	status, err := cloudStatus(ctx)
	if err != nil {
		slog.Debug("could not check whether Ollama cloud is enabled", "error", err)
	} else if status != nil && status.Cloud.Disabled {
		return false
	}

	user, err := whoami(ctx)
	if err != nil {
		var authErr api.AuthorizationError
		if errors.As(err, &authErr) && authErr.StatusCode == http.StatusUnauthorized {
			return false
		}
		slog.Debug("could not check whether cloud models are available", "error", err)
		return true
	}
	return user != nil && strings.TrimSpace(user.Name) != ""
}

func claudeLocalModels(
	ctx context.Context,
	list func(context.Context) (*api.ListResponse, error),
) ([]string, error) {
	ctx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	response, err := list(ctx)
	if err != nil || response == nil {
		return nil, err
	}

	names := make([]string, 0, len(response.Models)*2)
	for _, model := range response.Models {
		if model.RemoteModel != "" || model.RemoteHost != "" {
			continue
		}
		if model.Name != "" {
			names = append(names, model.Name)
		}
		if model.Model != "" && model.Model != model.Name {
			names = append(names, model.Model)
		}
	}
	return names, nil
}

func claudeGatewayPort() (string, error) {
	_, port, err := net.SplitHostPort(claudeProxyListenAddr)
	if err != nil {
		return "", err
	}
	return port, nil
}

//export ClaudeGatewayPort
func ClaudeGatewayPort() C.int {
	portText, err := claudeGatewayPort()
	if err != nil {
		return 0
	}
	port, err := strconv.Atoi(portText)
	if err != nil {
		return 0
	}
	return C.int(port)
}

// These helpers require claudeProxyMu to be held.
func setClaudeProxyFailure(err error, failure claudeProxyFailure) error {
	claudeProxyErr = err
	claudeProxyFail = failure
	return err
}

func clearClaudeProxyFailure() {
	claudeProxyErr = nil
	claudeProxyFail = claudeProxyFailureNone
}

//export SetClaudeGatewayInstalled
func SetClaudeGatewayInstalled(installed C.bool, restartClaude C.bool) C.bool {
	shouldInstall := installed != C._Bool(false)
	shouldRestart := restartClaude != C._Bool(false)
	if err := setClaudeGatewayInstalled(shouldInstall, shouldRestart); err != nil {
		slog.Warn("failed to change Claude Desktop gateway installation", "installed", shouldInstall, "error", err)
		return C._Bool(false)
	}
	return C._Bool(true)
}

func setClaudeGatewayInstalled(installed, restart bool) error {
	if installed && !claudeDesktopInstalled() {
		return errors.New("Claude Desktop is not installed")
	}
	if installed {
		if err := startClaudeAppProxy(); err != nil {
			return err
		}
	}
	err := claudeDesktop.SetInstalledFromDesktop(installed, restart)
	if !claudeDesktop.UsesOllamaGateway() {
		stopClaudeAppProxy()
	}
	return err
}

//export IsClaudeDesktopInstalled
func IsClaudeDesktopInstalled() C.bool {
	return C._Bool(claudeDesktopInstalled())
}

//export IsClaudeDesktopRunning
func IsClaudeDesktopRunning() C.bool {
	return C._Bool(launch.ClaudeDesktopRunning())
}

//export IsClaudeGatewayConfigured
func IsClaudeGatewayConfigured() C.bool {
	return C._Bool(claudeDesktop.UsesOllamaGateway())
}

//export ClaudeGatewayStartFailed
func ClaudeGatewayStartFailed() C.bool {
	return C._Bool(claudeGatewayStartFailed())
}

func claudeGatewayStartFailed() bool {
	claudeProxyMu.Lock()
	defer claudeProxyMu.Unlock()
	return claudeProxyErr != nil
}

//export ClaudeGatewayPortConflict
func ClaudeGatewayPortConflict() C.bool {
	return C._Bool(claudeGatewayPortConflict())
}

func claudeGatewayPortConflict() bool {
	claudeProxyMu.Lock()
	defer claudeProxyMu.Unlock()
	return claudeProxyFail == claudeProxyFailurePortConflict
}

func getClaudeDesktopConnectionStatus() claudeDesktopStatus {
	claudeProxyMu.Lock()
	proxyErr := claudeProxyErr
	proxyFailure := claudeProxyFail
	claudeProxyMu.Unlock()

	port := 0
	if portText, err := claudeGatewayPort(); err == nil {
		port, _ = strconv.Atoi(portText)
	}
	installed := claudeDesktopInstalled()
	configured := installed && claudeDesktop.UsesOllamaGateway()
	status := claudeDesktopStatus{
		Supported:    true,
		Installed:    installed,
		Connected:    configured && proxyErr == nil,
		Running:      launch.ClaudeDesktopRunning(),
		StartFailed:  proxyErr != nil,
		PortConflict: proxyFailure == claudeProxyFailurePortConflict,
		GatewayPort:  port,
	}
	if proxyErr != nil {
		status.Error = proxyErr.Error()
	}
	return status
}

func setClaudeDesktopConnection(enabled bool) error {
	if !claudeDesktopInstalled() {
		return errors.New("Claude Desktop is not installed")
	}
	return setClaudeGatewayInstalled(enabled, launch.ClaudeDesktopRunning())
}

func openClaudeDesktopApplication() error {
	return launch.OpenClaudeDesktop()
}

func requestClaudeDesktopInstall() claudeDesktopInstallResult {
	return claudeDesktopInstallResultFromCode(int(C.installClaudeDesktop()))
}

func getShowAppsInMenu() bool {
	return bool(C.ShowAppsInMenu())
}

func setShowAppsInMenu(visible bool) {
	C.SetShowAppsInMenu(C._Bool(visible))
}

func claudeDesktopInstallResultFromCode(code int) claudeDesktopInstallResult {
	switch code {
	case int(C.ClaudeInstallerOpened):
		return claudeDesktopInstallerOpened
	case int(C.ClaudeInstallCancelled):
		return claudeDesktopInstallCancelled
	default:
		return claudeDesktopInstallFailed
	}
}

//export RefreshClaudeProxyMenu
func RefreshClaudeProxyMenu() {
	claudeProxyMu.Lock()
	proxy := claudeAppProxy
	claudeProxyMu.Unlock()
	if proxy == nil {
		return
	}
	updateClaudeProxyMenu(proxy.Counts())
}

func updateClaudeProxyMenu(counts proxy.ClaudeDesktopCounts) {
	C.updateClaudeProxyMenu(C.ulonglong(counts.Routed))
}

func stopClaudeAppProxy() {
	claudeProxyMu.Lock()
	proxy := claudeAppProxy
	claudeAppProxy = nil
	clearClaudeProxyFailure()
	claudeProxyMu.Unlock()
	if proxy == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := proxy.Close(ctx); err != nil {
		slog.Debug("failed to stop Claude gateway cleanly", "error", err)
	}
}

func quit() {
	ctx, cancel := context.WithTimeout(context.Background(), claudeShutdownTimeout)
	defer cancel()
	handoff := bool(C.otherOllamaInstanceRunning())
	if err := restoreClaudeAppForTermination(ctx, handoff); err != nil {
		slog.Warn("failed to restore Claude before quitting", "error", err)
	}
	C.quit()
}

func restoreClaudeBeforeQuit(ctx context.Context, handoff, configured bool, restore func(context.Context) error) error {
	if handoff || !configured {
		return nil
	}
	return restore(ctx)
}

func restoreClaudeAppForTermination(ctx context.Context, handoff bool) error {
	if handoff {
		stopClaudeAppProxy()
		return nil
	}
	configured := claudeDesktop.UsesOllamaGateway()
	err := restoreClaudeBeforeQuit(ctx, handoff, configured, claudeDesktop.RestoreForShutdown)
	if !claudeDesktop.UsesOllamaGateway() {
		stopClaudeAppProxy()
	}
	return err
}

//export RestoreClaudeGatewayForShutdown
func RestoreClaudeGatewayForShutdown() C.bool {
	ctx, cancel := context.WithTimeout(context.Background(), claudeShutdownTimeout)
	defer cancel()
	if err := restoreClaudeAppForTermination(ctx, false); err != nil {
		slog.Warn("failed to restore Claude during system shutdown", "error", err)
		return C._Bool(false)
	}
	return C._Bool(true)
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

func setOnboardingWindowStyle(ptr unsafe.Pointer, enabled bool) {
	styleWindow(ptr)
	C.setWindowResizable(C.uintptr_t(uintptr(ptr)), C.bool(!enabled))
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
func checkAndHandleExistingInstance(_ string) {}
