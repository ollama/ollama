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
	appui "github.com/ollama/ollama/app/ui"
	"github.com/ollama/ollama/app/updater"
	"github.com/ollama/ollama/app/version"
	"github.com/ollama/ollama/cmd/launch"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/modelref"
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

// claudeDesktopController abstracts launch's Claude Desktop profile management
// so app flows can be tested without probing a live gateway.
type claudeDesktopController interface {
	UsesOllamaGateway() bool
	SetInstalledFromDesktop(installed, restart bool) error
	RestoreForShutdown(ctx context.Context) error
}

var (
	isApp           = updater.BundlePath != ""
	appLogPath      = filepath.Join(os.Getenv("HOME"), ".ollama", "logs", "app.log")
	launchAgentPath = filepath.Join(os.Getenv("HOME"), "Library", "LaunchAgents", "com.ollama.ollama.plist")
	claudeAppProxy  *proxy.ClaudeDesktop
	claudeProxyMu   sync.Mutex
	claudeProxyErr  error
	claudeProxyFail claudeProxyFailure
	claudeDesktop   claudeDesktopController = &launch.ClaudeDesktop{}

	claudeDesktopInstalled        = launch.ClaudeDesktopInstalled
	claudeProxyListenAddr         = proxy.DefaultClaudeDesktopListenAddr
	claudeProxyRetryWait          = 750 * time.Millisecond
	claudeProxyRetryPoll          = 50 * time.Millisecond
	claudeAccessRetryWait         = 3 * time.Second
	claudeAccessRetryPoll         = 100 * time.Millisecond
	claudeShutdownTimeout         = 30 * time.Second
	claudeRecommendationsClient   = &http.Client{Timeout: 3 * time.Second}
	claudeRecommendationsEndpoint = func() string {
		return strings.TrimRight(appui.OllamaDotCom, "/") + "/api/experimental/model-recommendations?app=claude-desktop"
	}
	claudeDownloadEndpoint = func() string {
		return claudeDesktopDownloadEndpoint(appui.OllamaDotCom)
	}
	claudeModelsLoader    = loadClaudeDesktopModels
	claudeAvailableModels []proxy.ClaudeDesktopModel
	claudeModelSource     string

	claudeAccessStateResolver = currentClaudeDesktopAccessState
	claudeLocalModelsResolver = currentClaudeDesktopLocalModels
)

var errClaudeDesktopAccessUnavailable = errors.New("Ollama couldn't verify the selected models. Try again")

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
	configured := claudeDesktop.UsesOllamaGateway()
	if configured || len(launch.ClaudeDesktopModels()) > 0 {
		if err := markClaudeDesktopIntegrationUsed(); err != nil {
			slog.Warn("failed to remember existing Claude Desktop connection", "error", err)
		}
	}
	if !claudeDesktopInstalled() || !configured {
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
	availableModels, activeModels, modelSource := resolveClaudeDesktopCatalog(context.Background())
	if err := ensureClaudeDesktopModelsAvailable(context.Background(), activeModels); err != nil {
		return setClaudeProxyFailure(err, claudeProxyFailureNone)
	}
	gateway, err := proxy.NewClaudeDesktop(proxy.ClaudeDesktopConfig{
		ListenAddr:         claudeProxyListenAddr,
		OllamaURL:          ollamaURL.String(),
		Model:              activeModels[0].OllamaModel,
		Models:             activeModels,
		Logger:             slog.Default(),
		OnCountsChanged:    updateClaudeProxyMenu,
		ResolveAccessState: claudeAccessStateResolver,
		ListLocalModels:    claudeLocalModelsResolver,
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
	claudeAvailableModels = availableModels
	claudeModelSource = modelSource
	clearClaudeProxyFailure()
	return nil
}

func resolveClaudeDesktopCatalog(ctx context.Context) (available, selected []proxy.ClaudeDesktopModel, source string) {
	available, source = claudeModelsLoader(ctx)
	selectedNames := launch.ClaudeDesktopModels()
	selected = proxy.SelectClaudeDesktopModels(available, selectedNames)
	if len(selected) == 0 {
		selected = available
	}
	available = includeSelectedClaudeDesktopModels(available, selected)
	if len(selectedNames) > 0 {
		source = "user"
	}
	return available, selected, source
}

func loadClaudeDesktopModels(ctx context.Context) ([]proxy.ClaudeDesktopModel, string) {
	models, err := proxy.FetchClaudeDesktopModels(ctx, claudeRecommendationsClient, claudeRecommendationsEndpoint())
	if err == nil {
		return models, "endpoint"
	}
	slog.Debug("could not fetch Claude Desktop model recommendations", "error", err)
	return proxy.DefaultClaudeDesktopModels(), "fallback"
}

func includeSelectedClaudeDesktopModels(available, selected []proxy.ClaudeDesktopModel) []proxy.ClaudeDesktopModel {
	models := append([]proxy.ClaudeDesktopModel(nil), available...)
	seen := make(map[string]struct{}, len(models))
	for _, model := range models {
		seen[model.Name] = struct{}{}
	}
	for _, model := range selected {
		if _, ok := seen[model.Name]; ok {
			continue
		}
		models = append(models, model)
		seen[model.Name] = struct{}{}
	}
	return models
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

func resolveClaudeDesktopAccessState(
	ctx context.Context,
	cloudStatus func(context.Context) (*api.StatusResponse, error),
	whoami func(context.Context) (*api.UserResponse, error),
) (proxy.ClaudeDesktopAccessState, error) {
	state := proxy.ClaudeDesktopAccessState{
		Cloud:   proxy.ClaudeDesktopCloudUnknown,
		Account: proxy.ClaudeDesktopAccountUnknown,
	}
	statusCtx, cancel := context.WithTimeout(ctx, time.Second)
	status, err := cloudStatus(statusCtx)
	cancel()
	if err != nil {
		return state, fmt.Errorf("check whether Ollama cloud is enabled: %w", err)
	}
	if status != nil && status.Cloud.Disabled {
		state.Cloud = proxy.ClaudeDesktopCloudOff
		// Account eligibility cannot change the result while Cloud is off, so do
		// not make a remote account request from discovery or Settings.
		return state, nil
	}
	state.Cloud = proxy.ClaudeDesktopCloudOn

	accountCtx, cancel := context.WithTimeout(ctx, time.Second)
	user, err := whoami(accountCtx)
	defer cancel()
	if err != nil {
		var authErr api.AuthorizationError
		if errors.As(err, &authErr) && authErr.StatusCode == http.StatusUnauthorized {
			state.Account = proxy.ClaudeDesktopAccountSignedOut
			return state, nil
		}
		return state, fmt.Errorf("check Ollama account: %w", err)
	}
	if user == nil || strings.TrimSpace(user.Name) == "" {
		state.Account = proxy.ClaudeDesktopAccountSignedOut
		return state, nil
	}
	state.Account = proxy.ClaudeDesktopAccountSignedIn
	state.Plan = strings.TrimSpace(user.Plan)
	return state, nil
}

func currentClaudeDesktopAccessState(ctx context.Context) (proxy.ClaudeDesktopAccessState, error) {
	client := api.NewClient(envconfig.ConnectableHost(), http.DefaultClient)
	return resolveClaudeDesktopAccessState(ctx, client.CloudStatusExperimental, client.Whoami)
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
		// /api/tags can include cloud placeholders. Only models with local
		// weights qualify as installed local choices for Claude Desktop.
		if model.RemoteModel != "" || model.RemoteHost != "" ||
			modelref.HasExplicitCloudSource(model.Name) || modelref.HasExplicitCloudSource(model.Model) {
			continue
		}
		if model.Name != "" {
			names = append(names, model.Name)
			if alias := strings.TrimSuffix(model.Name, ":latest"); alias != model.Name {
				names = append(names, alias)
			}
		}
		if model.Model != "" && model.Model != model.Name {
			names = append(names, model.Model)
			if alias := strings.TrimSuffix(model.Model, ":latest"); alias != model.Model {
				names = append(names, alias)
			}
		}
	}
	return names, nil
}

func currentClaudeDesktopLocalModels(ctx context.Context) ([]string, error) {
	client := api.NewClient(envconfig.ConnectableHost(), http.DefaultClient)
	return claudeLocalModels(ctx, client.List)
}

func ensureClaudeDesktopModelsAvailable(ctx context.Context, models []proxy.ClaudeDesktopModel) error {
	deadline := time.Now().Add(claudeAccessRetryWait)
	for {
		state, stateErr := claudeAccessStateResolver(ctx)
		if stateErr != nil {
			slog.Debug("could not resolve Claude model access", "error", stateErr)
		}
		localNames, localErr := claudeLocalModelsResolver(ctx)
		if localErr != nil {
			slog.Debug("could not load local models for Claude", "error", localErr)
		}
		err := validateClaudeDesktopModels(models, state, localNames, localErr == nil)
		if !errors.Is(err, errClaudeDesktopAccessUnavailable) || time.Now().After(deadline) {
			return err
		}

		timer := time.NewTimer(claudeAccessRetryPoll)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
}

func validateClaudeDesktopModels(models []proxy.ClaudeDesktopModel, state proxy.ClaudeDesktopAccessState, localNames []string, inventoryKnown bool) error {
	installed := make(map[string]struct{}, len(localNames))
	for _, name := range localNames {
		installed[name] = struct{}{}
	}

	reasons := make(map[proxy.ClaudeDesktopAccessReason]struct{})
	for _, model := range models {
		_, isInstalled := installed[model.OllamaModel]
		access := proxy.EvaluateClaudeDesktopModelAccess(model, state, isInstalled, inventoryKnown)
		if access.Availability == proxy.ClaudeDesktopAvailabilityAvailable {
			return nil
		}
		reasons[access.Reason] = struct{}{}
	}

	// Prefer the action that resolves the broadest part of the selected set.
	if _, ok := reasons[proxy.ClaudeDesktopAccessCloudOff]; ok {
		return errors.New("Cloud models are off. Choose an installed model in Ollama Settings")
	}
	if _, ok := reasons[proxy.ClaudeDesktopAccessSignInRequired]; ok {
		return errors.New("Sign in to Ollama or choose an installed model in Ollama Settings")
	}
	if _, ok := reasons[proxy.ClaudeDesktopAccessUpgradeRequired]; ok {
		return errors.New("Select another model in Settings to connect Claude")
	}
	if _, ok := reasons[proxy.ClaudeDesktopAccessModelNotInstalled]; ok {
		return errors.New("Install the selected model or choose another model in Ollama Settings")
	}
	if _, ok := reasons[proxy.ClaudeDesktopAccessVerificationUnavailable]; ok {
		return errClaudeDesktopAccessUnavailable
	}
	return errors.New("Choose at least one model in Ollama Settings")
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

//export HasUsedClaudeDesktopIntegration
func HasUsedClaudeDesktopIntegration() C.bool {
	return C._Bool(hasUsedClaudeDesktopIntegration())
}

func hasUsedClaudeDesktopIntegration() bool {
	if appStore == nil {
		return false
	}
	settings, err := appStore.Settings()
	if err != nil {
		slog.Warn("failed to read Claude Desktop integration history", "error", err)
		return false
	}
	return settings.ClaudeDesktopUsed
}

func markClaudeDesktopIntegrationUsed() error {
	if appStore == nil {
		return nil
	}
	settings, err := appStore.Settings()
	if err != nil {
		return fmt.Errorf("load settings: %w", err)
	}
	if settings.ClaudeDesktopUsed {
		return nil
	}
	settings.ClaudeDesktopUsed = true
	if err := appStore.SetSettings(settings); err != nil {
		return fmt.Errorf("save settings: %w", err)
	}
	return nil
}

func setClaudeGatewayInstalled(installed, restart bool) error {
	if installed && !claudeDesktopInstalled() {
		return errors.New("Claude Desktop is not installed")
	}
	if installed {
		// A failed first attempt must still expose Claude's Settings recovery UI.
		if err := markClaudeDesktopIntegrationUsed(); err != nil {
			return fmt.Errorf("remember Claude Desktop connection: %w", err)
		}
		if err := startClaudeAppProxy(); err != nil {
			return err
		}
	}
	err := claudeDesktop.SetInstalledFromDesktop(installed, restart)
	if !claudeDesktop.UsesOllamaGateway() {
		stopClaudeAppProxy()
	}
	if err != nil {
		return err
	}
	return nil
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

//export ClaudeGatewayErrorMessage
func ClaudeGatewayErrorMessage() *C.char {
	claudeProxyMu.Lock()
	defer claudeProxyMu.Unlock()
	if claudeProxyErr == nil {
		return C.CString("")
	}
	return C.CString(claudeProxyErr.Error())
}

func claudeGatewayPortConflict() bool {
	claudeProxyMu.Lock()
	defer claudeProxyMu.Unlock()
	return claudeProxyFail == claudeProxyFailurePortConflict
}

func getClaudeDesktopConnectionStatus() claudeDesktopStatus {
	used := hasUsedClaudeDesktopIntegration()
	claudeProxyMu.Lock()
	proxyErr := claudeProxyErr
	proxyFailure := claudeProxyFail
	gateway := claudeAppProxy
	availableModels := append([]proxy.ClaudeDesktopModel(nil), claudeAvailableModels...)
	modelSource := claudeModelSource
	claudeProxyMu.Unlock()
	var selectedModels []proxy.ClaudeDesktopModel
	if gateway != nil {
		selectedModels = gateway.Models()
	} else if used {
		if len(availableModels) == 0 {
			availableModels, selectedModels, modelSource = resolveClaudeDesktopCatalog(context.Background())
			claudeProxyMu.Lock()
			if len(claudeAvailableModels) == 0 {
				claudeAvailableModels = append([]proxy.ClaudeDesktopModel(nil), availableModels...)
				claudeModelSource = modelSource
			}
			claudeProxyMu.Unlock()
		} else {
			selectedModels = proxy.SelectClaudeDesktopModels(availableModels, launch.ClaudeDesktopModels())
			if len(selectedModels) == 0 {
				selectedModels = availableModels
			}
		}
	}
	selected := make(map[string]struct{})
	for _, model := range selectedModels {
		selected[model.Name] = struct{}{}
	}
	accessState := proxy.ClaudeDesktopAccessState{
		Cloud:   proxy.ClaudeDesktopCloudUnknown,
		Account: proxy.ClaudeDesktopAccountUnknown,
	}
	var localNames []string
	var localErr error
	if len(availableModels) > 0 {
		var accessErr error
		accessState, accessErr = claudeAccessStateResolver(context.Background())
		if accessErr != nil {
			slog.Debug("could not resolve Claude model access for Settings", "error", accessErr)
		}
		localNames, localErr = claudeLocalModelsResolver(context.Background())
		if localErr != nil {
			slog.Debug("could not load local models for Claude Settings", "error", localErr)
		}
	}
	localModels := make(map[string]struct{}, len(localNames))
	for _, name := range localNames {
		localModels[name] = struct{}{}
	}
	modelStatuses := make([]claudeDesktopModelStatus, 0, len(availableModels))
	for _, model := range availableModels {
		_, isSelected := selected[model.Name]
		name := model.Name
		if model.Cloud {
			name = model.OllamaModel
		}
		_, installed := localModels[model.OllamaModel]
		access := proxy.EvaluateClaudeDesktopModelAccess(model, accessState, installed, localErr == nil)
		modelStatuses = append(modelStatuses, claudeDesktopModelStatus{
			Name:         name,
			DisplayName:  name,
			Description:  model.Description,
			Cloud:        model.Cloud,
			Selected:     isSelected,
			Availability: access.Availability,
			Reason:       access.Reason,
			RequiredPlan: access.RequiredPlan,
		})
	}

	port := 0
	if portText, err := claudeGatewayPort(); err == nil {
		port, _ = strconv.Atoi(portText)
	}
	installed := claudeDesktopInstalled()
	configured := installed && claudeDesktop.UsesOllamaGateway()
	status := claudeDesktopStatus{
		Supported:    true,
		Used:         used,
		Installed:    installed,
		Connected:    configured && proxyErr == nil,
		Running:      launch.ClaudeDesktopRunning(),
		StartFailed:  proxyErr != nil,
		PortConflict: proxyFailure == claudeProxyFailurePortConflict,
		GatewayPort:  port,
		ModelSource:  modelSource,
		Models:       modelStatuses,
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

func restartClaudeDesktopWithModels(names []string) error {
	if !claudeDesktopInstalled() {
		return errors.New("Claude Desktop is not installed")
	}
	if len(names) == 0 {
		return errors.New("select at least one Claude Desktop model")
	}
	if len(names) > proxy.MaxClaudeDesktopModels {
		return fmt.Errorf("Claude Desktop supports at most %d models; deselect %d and try again", proxy.MaxClaudeDesktopModels, len(names)-proxy.MaxClaudeDesktopModels)
	}
	claudeProxyMu.Lock()
	gateway := claudeAppProxy
	available := append([]proxy.ClaudeDesktopModel(nil), claudeAvailableModels...)
	claudeProxyMu.Unlock()
	var current []proxy.ClaudeDesktopModel
	if gateway != nil {
		current = gateway.Models()
	}
	if len(available) == 0 {
		var modelSource string
		available, current, modelSource = resolveClaudeDesktopCatalog(context.Background())
		claudeProxyMu.Lock()
		claudeAvailableModels = append([]proxy.ClaudeDesktopModel(nil), available...)
		claudeModelSource = modelSource
		claudeProxyMu.Unlock()
	} else if len(current) == 0 {
		current = proxy.SelectClaudeDesktopModels(available, launch.ClaudeDesktopModels())
		if len(current) == 0 {
			current = available
		}
	}

	localNames, localErr := claudeLocalModelsResolver(context.Background())
	if localErr != nil {
		slog.Debug("could not load local models for Claude Desktop selection", "error", localErr)
	}
	selected, err := selectKnownClaudeDesktopModels(available, current, localNames, names)
	if err != nil {
		return err
	}
	accessState, accessErr := claudeAccessStateResolver(context.Background())
	if accessErr != nil {
		slog.Debug("could not resolve Claude model access for selection", "error", accessErr)
	}
	if err := validateClaudeDesktopModels(selected, accessState, localNames, localErr == nil); err != nil {
		return err
	}

	normalized := make([]string, len(selected))
	for i, model := range selected {
		normalized[i] = model.Name
		if model.Cloud {
			normalized[i] = model.OllamaModel
		}
	}
	if err := launch.SaveClaudeDesktopModels(normalized); err != nil {
		return fmt.Errorf("save Claude Desktop models: %w", err)
	}
	if gateway == nil || !claudeDesktop.UsesOllamaGateway() {
		claudeProxyMu.Lock()
		claudeAvailableModels = includeSelectedClaudeDesktopModels(available, selected)
		claudeModelSource = "user"
		claudeProxyMu.Unlock()
		return setClaudeGatewayInstalled(true, launch.ClaudeDesktopRunning())
	}
	if err := gateway.SetModels(selected); err != nil {
		return err
	}
	claudeProxyMu.Lock()
	claudeAvailableModels = includeSelectedClaudeDesktopModels(available, selected)
	claudeModelSource = "user"
	claudeProxyMu.Unlock()
	return claudeDesktop.SetInstalledFromDesktop(true, true)
}

func selectKnownClaudeDesktopModels(available, current []proxy.ClaudeDesktopModel, localNames, names []string) ([]proxy.ClaudeDesktopModel, error) {
	selectable := includeSelectedClaudeDesktopModels(available, current)
	allowed := make(map[string]struct{}, len(selectable)+len(localNames))
	for _, model := range selectable {
		allowed[model.Name] = struct{}{}
		allowed[model.OllamaModel] = struct{}{}
	}
	for _, name := range localNames {
		allowed[strings.TrimSpace(name)] = struct{}{}
	}
	for _, name := range names {
		if _, ok := allowed[strings.TrimSpace(name)]; !ok {
			return nil, fmt.Errorf("model %q is not installed or recommended for Claude Desktop", name)
		}
	}

	selected := proxy.SelectClaudeDesktopModels(selectable, names)
	if len(selected) == 0 {
		return nil, errors.New("select at least one Claude Desktop model")
	}
	return selected, nil
}

func requestClaudeDesktopInstall() claudeDesktopInstallResult {
	return claudeDesktopInstallResultFromCode(int(C.installClaudeDesktop()))
}

func claudeDesktopDownloadEndpoint(baseURL string) string {
	return strings.TrimRight(baseURL, "/") + "/download-app?app=claude-desktop&type=mac-zip"
}

//export ClaudeDesktopDownloadURL
func ClaudeDesktopDownloadURL() *C.char {
	return C.CString(claudeDownloadEndpoint())
}

//export InstallClaudeDesktopArchive
func InstallClaudeDesktopArchive(path *C.cchar_t) C.bool {
	archivePath := C.GoString((*C.char)(unsafe.Pointer(path)))
	installedPath, err := installClaudeDesktopZip(archivePath, claudeDesktopInstallDestinations(), verifyClaudeDesktopBundle)
	if errors.Is(err, errClaudeDesktopDestinationExists) && claudeDesktopInstalled() {
		slog.Info("Claude Desktop was installed while its download was in progress")
		return C._Bool(true)
	}
	if err != nil {
		slog.Warn("failed to install Claude Desktop archive", "error", err)
		return C._Bool(false)
	}
	slog.Info("installed Claude Desktop", "path", installedPath)
	return C._Bool(true)
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
