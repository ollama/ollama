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
	ollamaAuth "github.com/ollama/ollama/auth"
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
	AutodiscoveryConfiguredWithAutoMode(autoMode bool) bool
	UsesOllamaGateway() bool
	ConfigureAutodiscoveryWithAutoMode(autoMode bool) error
	SetInstalledFromDesktopWithAutoMode(installed, restart, autoMode bool) error
	RestartWithProfileChange(change func() error) error
	RestoreForShutdown(ctx context.Context) error
}

var (
	isApp              = updater.BundlePath != ""
	appLogPath         = filepath.Join(os.Getenv("HOME"), ".ollama", "logs", "app.log")
	launchAgentPath    = filepath.Join(os.Getenv("HOME"), "Library", "LaunchAgents", "com.ollama.ollama.plist")
	claudeAppProxy     *proxy.ClaudeDesktop
	claudeProxyStartMu sync.Mutex
	claudeProxyMu      sync.Mutex
	claudeCatalogMu    sync.Mutex
	claudeProxyErr     error
	claudeProxyFail    claudeProxyFailure
	claudeDesktop      claudeDesktopController = &launch.ClaudeDesktop{}

	claudeDesktopInstalled        = launch.ClaudeDesktopInstalled
	claudeDesktopRunning          = launch.ClaudeDesktopRunning
	claudeProxyListenAddr         = proxy.DefaultClaudeDesktopListenAddr
	claudeProxyRetryWait          = 750 * time.Millisecond
	claudeProxyRetryPoll          = 50 * time.Millisecond
	claudeAccessRetryWait         = 3 * time.Second
	claudeAccessRetryPoll         = 100 * time.Millisecond
	claudeCatalogRefreshInterval  = time.Minute
	claudeCatalogNow              = time.Now
	claudeShutdownTimeout         = 30 * time.Second
	claudeRecommendationsClient   = &http.Client{Timeout: 3 * time.Second}
	claudeRecommendationsEndpoint = func() string {
		return strings.TrimRight(appui.OllamaDotCom, "/") + "/api/experimental/model-recommendations?app=claude-desktop"
	}
	signOllamaData        = ollamaAuth.Sign
	claudeModelsLoader    = loadClaudeDesktopModels
	claudeAvailableModels []proxy.ClaudeDesktopModel
	claudeModelSource     string
	claudeCatalogUpdated  time.Time

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

//export IsOnboardingActive
func IsOnboardingActive() C.bool {
	return C._Bool(wv.OnboardingActive())
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
	if err := startClaudeAppProxy(); err != nil {
		return err
	}
	return updateClaudeDesktopProfile()
}

func updateClaudeDesktopProfile() error {
	autoMode, err := effectiveClaudeDesktopAutoMode(activeClaudeDesktopModels())
	if err != nil {
		return err
	}
	if claudeDesktop.AutodiscoveryConfiguredWithAutoMode(autoMode) {
		return nil
	}
	// Claude writes settings during shutdown. Do not race that write during
	// startup reconciliation; explicit settings changes use the restart path.
	if claudeDesktopRunning() {
		return nil
	}
	if err := claudeDesktop.ConfigureAutodiscoveryWithAutoMode(autoMode); err != nil {
		return fmt.Errorf("update Claude Desktop profile: %w", err)
	}
	return nil
}

func startClaudeAppProxy() error {
	claudeProxyStartMu.Lock()
	defer claudeProxyStartMu.Unlock()
	claudeProxyMu.Lock()
	if claudeAppProxy != nil {
		clearClaudeProxyFailure()
		claudeProxyMu.Unlock()
		return nil
	}
	claudeProxyMu.Unlock()
	if !claudeDesktopInstalled() {
		slog.Debug("Claude Desktop is not installed; skipping gateway")
		claudeProxyMu.Lock()
		clearClaudeProxyFailure()
		claudeProxyMu.Unlock()
		return nil
	}
	ollamaURL := envconfig.ConnectableHost()
	gatewayPort, err := claudeGatewayPort()
	if err != nil {
		return recordClaudeProxyFailure(fmt.Errorf("parse Claude gateway address: %w", err), claudeProxyFailureNone)
	}
	if ollamaURL.Port() == gatewayPort {
		return recordClaudeProxyFailure(
			fmt.Errorf("OLLAMA_HOST cannot use port %s because it is reserved for Claude", gatewayPort),
			claudeProxyFailurePortConflict,
		)
	}
	availableModels, activeModels, modelSource := resolveClaudeDesktopStartupCatalog(context.Background())
	if err := ensureClaudeDesktopModelsAvailable(context.Background(), activeModels); err != nil {
		return recordClaudeProxyFailure(err, claudeProxyFailureNone)
	}
	gateway, err := proxy.NewClaudeDesktop(proxy.ClaudeDesktopConfig{
		ListenAddr:      claudeProxyListenAddr,
		OllamaURL:       ollamaURL.String(),
		Model:           activeModels[0].OllamaModel,
		Models:          activeModels,
		Logger:          slog.Default(),
		OnCountsChanged: updateClaudeProxyMenu,
		RefreshModels: func(ctx context.Context, current []proxy.ClaudeDesktopModel) ([]proxy.ClaudeDesktopModel, error) {
			_, selected, _ := refreshClaudeDesktopCatalog(ctx, current, false)
			return selected, nil
		},
		ResolveAccessState: claudeAccessStateResolver,
		ListLocalModels:    claudeLocalModelsResolver,
	})
	if err != nil {
		return recordClaudeProxyFailure(err, claudeProxyFailureNone)
	}
	if err := startClaudeGateway(gateway); err != nil {
		_ = gateway.Close(context.Background())
		failure := claudeProxyFailureNone
		if errors.Is(err, syscall.EADDRINUSE) {
			failure = claudeProxyFailurePortConflict
		}
		return recordClaudeProxyFailure(err, failure)
	}
	claudeProxyMu.Lock()
	claudeAppProxy = gateway
	claudeAvailableModels = availableModels
	claudeModelSource = modelSource
	claudeCatalogUpdated = claudeCatalogNow()
	clearClaudeProxyFailure()
	claudeProxyMu.Unlock()
	return nil
}

func resolveClaudeDesktopStartupCatalog(ctx context.Context) (available, selected []proxy.ClaudeDesktopModel, source string) {
	selectedNames := launch.ClaudeDesktopModels()
	if len(selectedNames) > 0 {
		localNames, err := claudeLocalModelsResolver(ctx)
		if err == nil && allClaudeDesktopModelsLocal(selectedNames, localNames) {
			selected = proxy.SelectClaudeDesktopModels(nil, selectedNames)
			return selected, selected, "user"
		}
	}
	return refreshClaudeDesktopCatalog(ctx, nil, true)
}

func allClaudeDesktopModelsLocal(selected, installed []string) bool {
	local := make(map[string]struct{}, len(installed))
	for _, name := range installed {
		local[strings.TrimSpace(name)] = struct{}{}
	}
	for _, name := range selected {
		name = strings.TrimSpace(name)
		if name == "" || modelref.HasExplicitCloudSource(name) {
			return false
		}
		if _, ok := local[name]; !ok {
			return false
		}
	}
	return true
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

func refreshClaudeDesktopCatalog(ctx context.Context, current []proxy.ClaudeDesktopModel, force bool) (available, selected []proxy.ClaudeDesktopModel, source string) {
	claudeCatalogMu.Lock()
	defer claudeCatalogMu.Unlock()
	claudeProxyMu.Lock()
	previous := append([]proxy.ClaudeDesktopModel(nil), claudeAvailableModels...)
	// A local-only startup deliberately skips catalog discovery. Do not treat
	// that selected-model snapshot as a complete catalog when Settings asks for
	// all available choices.
	fresh := hasCloudClaudeDesktopModel(claudeAvailableModels) && claudeCatalogNow().Before(claudeCatalogUpdated.Add(claudeCatalogRefreshInterval))
	if !force && fresh {
		available = append([]proxy.ClaudeDesktopModel(nil), claudeAvailableModels...)
		source = claudeModelSource
	}
	claudeProxyMu.Unlock()

	reloaded := len(available) == 0
	if reloaded {
		available, source = claudeModelsLoader(ctx)
		if source == "fallback" && len(previous) > 0 {
			available = preserveClaudeDesktopEntitlements(available, previous)
		}
	}
	selectedNames := launch.ClaudeDesktopModels()
	if len(current) > 0 {
		selectedNames = make([]string, len(current))
		for i, model := range current {
			selectedNames[i] = model.Name
		}
	}
	selected = proxy.SelectClaudeDesktopModels(available, selectedNames)
	if len(selected) == 0 {
		selected = available
	}
	available = includeSelectedClaudeDesktopModels(available, selected)
	if len(launch.ClaudeDesktopModels()) > 0 {
		source = "user"
	}

	claudeProxyMu.Lock()
	claudeAvailableModels = append([]proxy.ClaudeDesktopModel(nil), available...)
	claudeModelSource = source
	if reloaded {
		claudeCatalogUpdated = claudeCatalogNow()
	}
	claudeProxyMu.Unlock()
	return available, selected, source
}

func preserveClaudeDesktopEntitlements(fallback, previous []proxy.ClaudeDesktopModel) []proxy.ClaudeDesktopModel {
	models := proxy.UnverifyClaudeDesktopCloudEntitlements(fallback)
	known := make(map[string]proxy.ClaudeDesktopModel, len(previous)*2)
	for _, model := range previous {
		known[model.Name] = model
		known[model.OllamaModel] = model
	}
	for i, model := range models {
		if prior, ok := known[model.Name]; ok {
			models[i] = prior
			continue
		}
		if prior, ok := known[model.OllamaModel]; ok {
			models[i] = prior
		}
	}
	return models
}

func loadClaudeDesktopModels(ctx context.Context) ([]proxy.ClaudeDesktopModel, string) {
	req, err := newSignedOllamaRequest(ctx, http.MethodGet, claudeRecommendationsEndpoint())
	if err != nil {
		slog.Debug("could not prepare Claude Desktop model recommendations request", "error", err)
		return fallbackClaudeDesktopModels(), "fallback"
	}
	models, err := proxy.FetchClaudeDesktopModels(claudeRecommendationsClient, req)
	if err == nil {
		return models, "endpoint"
	}
	slog.Debug("could not fetch Claude Desktop model recommendations", "error", err)
	return fallbackClaudeDesktopModels(), "fallback"
}

func fallbackClaudeDesktopModels() []proxy.ClaudeDesktopModel {
	models := proxy.UnverifyClaudeDesktopCloudEntitlements(proxy.DefaultClaudeDesktopModels())
	for i := range models {
		models[i].Recommended = false
	}
	return models
}

func claudeDesktopModelSupportsAutoMode(model proxy.ClaudeDesktopModel) bool {
	if !model.Recommended {
		return false
	}
	for _, name := range []string{model.Name, model.OllamaModel} {
		name = strings.ToLower(strings.TrimSpace(name))
		if name == "gemma4" || strings.HasPrefix(name, "gemma4:") {
			return false
		}
	}
	return true
}

func claudeDesktopModelsSupportAutoMode(models []proxy.ClaudeDesktopModel) bool {
	for _, model := range models {
		if claudeDesktopModelSupportsAutoMode(model) {
			return true
		}
	}
	return false
}

func activeClaudeDesktopModels() []proxy.ClaudeDesktopModel {
	claudeProxyMu.Lock()
	gateway := claudeAppProxy
	available := append([]proxy.ClaudeDesktopModel(nil), claudeAvailableModels...)
	claudeProxyMu.Unlock()
	if gateway != nil {
		if models := gateway.Models(); len(models) > 0 {
			return models
		}
	}
	selected := proxy.SelectClaudeDesktopModels(available, launch.ClaudeDesktopModels())
	if len(selected) > 0 {
		return selected
	}
	return available
}

func effectiveClaudeDesktopAutoMode(models []proxy.ClaudeDesktopModel) (bool, error) {
	enabled, err := launch.ClaudeDesktopAutoModeEnabled()
	if err != nil {
		return false, err
	}
	return enabled && claudeDesktopModelsSupportAutoMode(models), nil
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
		state := proxy.ClaudeDesktopAccessState{}
		if hasCloudClaudeDesktopModel(models) {
			var stateErr error
			state, stateErr = claudeAccessStateResolver(ctx)
			if stateErr != nil {
				slog.Debug("could not resolve Claude model access", "error", stateErr)
			}
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

func hasCloudClaudeDesktopModel(models []proxy.ClaudeDesktopModel) bool {
	for _, model := range models {
		if model.Cloud {
			return true
		}
	}
	return false
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

func recordClaudeProxyFailure(err error, failure claudeProxyFailure) error {
	claudeProxyMu.Lock()
	defer claudeProxyMu.Unlock()
	return setClaudeProxyFailure(err, failure)
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
	autoMode := false
	if installed {
		var err error
		autoMode, err = effectiveClaudeDesktopAutoMode(activeClaudeDesktopModels())
		if err != nil {
			return err
		}
	}
	err := claudeDesktop.SetInstalledFromDesktopWithAutoMode(installed, restart, autoMode)
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

func getClaudeDesktopConnectionSummary() claudeDesktopStatus {
	return claudeDesktopConnectionSummary(hasUsedClaudeDesktopIntegration())
}

func claudeDesktopRequestCount() uint64 {
	claudeProxyMu.Lock()
	gateway := claudeAppProxy
	claudeProxyMu.Unlock()
	if gateway == nil {
		return 0
	}
	return gateway.Counts().Routed
}

func claudeDesktopConnectionSummary(used bool) claudeDesktopStatus {
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
		Supported:      true,
		Used:           used,
		Installed:      installed,
		Configured:     configured,
		Connected:      configured && proxyErr == nil,
		Running:        launch.ClaudeDesktopRunning(),
		StartFailed:    proxyErr != nil,
		PortConflict:   proxyFailure == claudeProxyFailurePortConflict,
		GatewayPort:    port,
		RoutedRequests: claudeDesktopRequestCount(),
	}
	if proxyErr != nil {
		status.Error = proxyErr.Error()
	}
	return status
}

func getClaudeDesktopConnectionStatus() claudeDesktopStatus {
	used := hasUsedClaudeDesktopIntegration()
	var availableModels, selectedModels []proxy.ClaudeDesktopModel
	var modelSource string
	if used {
		claudeProxyMu.Lock()
		gateway := claudeAppProxy
		claudeProxyMu.Unlock()
		var current []proxy.ClaudeDesktopModel
		if gateway != nil {
			current = gateway.Models()
		}
		// Settings needs the complete catalog even when Claude is currently
		// configured with only local models. The startup path can skip cloud
		// discovery, but reusing that local-only list here would make every cloud
		// choice disappear from the model picker.
		availableModels, selectedModels, modelSource = refreshClaudeDesktopCatalog(context.Background(), current, false)
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
	if hasCloudClaudeDesktopModel(availableModels) {
		var accessErr error
		accessState, accessErr = claudeAccessStateResolver(context.Background())
		if accessErr != nil {
			slog.Debug("could not resolve Claude model access for Settings", "error", accessErr)
		}
	}
	if len(availableModels) > 0 {
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
			AutoMode:     claudeDesktopModelSupportsAutoMode(model),
			Availability: access.Availability,
			Reason:       access.Reason,
			RequiredPlan: access.RequiredPlan,
		})
	}

	status := claudeDesktopConnectionSummary(used)
	autoMode, autoModeErr := launch.ClaudeDesktopAutoModeEnabled()
	status.AutoMode = autoMode
	status.ModelSource = modelSource
	status.Models = modelStatuses
	if status.Error == "" && autoModeErr != nil {
		status.Error = autoModeErr.Error()
	}
	return status
}

func setClaudeDesktopConnection(enabled, restartConfirmed bool) error {
	if !claudeDesktopInstalled() {
		return errors.New("Claude Desktop is not installed")
	}
	return setClaudeGatewayInstalled(enabled, restartConfirmed)
}

func prepareClaudeDesktopConnection() error {
	if !claudeDesktopInstalled() {
		return errors.New("Claude Desktop is not installed")
	}
	if err := startClaudeAppProxy(); err != nil {
		return err
	}
	autoMode, err := effectiveClaudeDesktopAutoMode(activeClaudeDesktopModels())
	if err == nil {
		err = claudeDesktop.ConfigureAutodiscoveryWithAutoMode(autoMode)
	}
	if !claudeDesktop.UsesOllamaGateway() {
		stopClaudeAppProxy()
	}
	return err
}

func openClaudeDesktopApplication() error {
	return launch.OpenClaudeDesktop()
}

func setClaudeDesktopAutoMode(enabled bool) error {
	models := activeClaudeDesktopModels()
	if enabled && !claudeDesktopModelsSupportAutoMode(models) {
		return errors.New("select at least one Auto-compatible recommended model in Claude settings")
	}
	previous, err := launch.ClaudeDesktopAutoModeEnabled()
	if err != nil {
		return err
	}
	if previous == enabled && (!claudeDesktop.UsesOllamaGateway() || claudeDesktop.AutodiscoveryConfiguredWithAutoMode(enabled)) {
		return nil
	}
	if err := launch.SaveClaudeDesktopAutoMode(enabled); err != nil {
		return fmt.Errorf("save Claude Desktop auto mode: %w", err)
	}
	if !claudeDesktop.UsesOllamaGateway() {
		// The preference takes effect the next time the profile is written.
		return nil
	}
	if claudeDesktopRunning() {
		return claudeDesktop.RestartWithProfileChange(func() error {
			return claudeDesktop.ConfigureAutodiscoveryWithAutoMode(enabled)
		})
	}
	return claudeDesktop.ConfigureAutodiscoveryWithAutoMode(enabled)
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
	previousAvailable := append([]proxy.ClaudeDesktopModel(nil), claudeAvailableModels...)
	previousSource := claudeModelSource
	previousCatalogUpdated := claudeCatalogUpdated
	claudeProxyMu.Unlock()
	var current []proxy.ClaudeDesktopModel
	if gateway != nil {
		current = gateway.Models()
	}
	available, refreshedCurrent, _ := refreshClaudeDesktopCatalog(context.Background(), current, true)
	if len(current) == 0 {
		current = refreshedCurrent
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
	previousSelection := launch.ClaudeDesktopModels()
	restoreState := func() error {
		var rollbackErr error
		if gateway != nil && len(current) > 0 {
			rollbackErr = gateway.SetModels(current)
		}
		if err := launch.RestoreClaudeDesktopModels(previousSelection); err != nil {
			rollbackErr = errors.Join(rollbackErr, fmt.Errorf("restore Claude Desktop model selection: %w", err))
		}
		claudeProxyMu.Lock()
		claudeAvailableModels = previousAvailable
		claudeModelSource = previousSource
		claudeCatalogUpdated = previousCatalogUpdated
		claudeProxyMu.Unlock()
		return rollbackErr
	}
	if gateway == nil || !claudeDesktop.UsesOllamaGateway() {
		if err := launch.SaveClaudeDesktopModels(normalized); err != nil {
			_ = restoreState()
			return fmt.Errorf("save Claude Desktop models: %w", err)
		}
		if err := setClaudeGatewayInstalled(true, launch.ClaudeDesktopRunning()); err != nil {
			// Profile installation can succeed even if opening Claude fails. Keep
			// the live and persisted model state aligned with that committed profile.
			if claudeDesktop.UsesOllamaGateway() {
				return err
			}
			return errors.Join(err, restoreState())
		}
		return nil
	}
	applyModelChange := func() error {
		if err := launch.SaveClaudeDesktopModels(normalized); err != nil {
			return errors.Join(fmt.Errorf("save Claude Desktop models: %w", err), restoreState())
		}
		if err := gateway.SetModels(selected); err != nil {
			return errors.Join(err, restoreState())
		}
		claudeProxyMu.Lock()
		claudeAvailableModels = includeSelectedClaudeDesktopModels(available, selected)
		claudeModelSource = "user"
		claudeProxyMu.Unlock()
		autoMode, err := effectiveClaudeDesktopAutoMode(selected)
		if err != nil {
			return errors.Join(err, restoreState())
		}
		if err := claudeDesktop.ConfigureAutodiscoveryWithAutoMode(autoMode); err != nil {
			return errors.Join(err, restoreState())
		}
		return nil
	}
	if err := claudeDesktop.RestartWithProfileChange(applyModelChange); err != nil {
		return errors.Join(err, restoreState())
	}
	return nil
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

func newSignedOllamaRequest(ctx context.Context, method, endpoint string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, endpoint, nil)
	if err != nil {
		return nil, err
	}
	query := req.URL.Query()
	query.Set("ts", strconv.FormatInt(time.Now().Unix(), 10))
	req.URL.RawQuery = query.Encode()
	signature, err := signOllamaData(ctx, []byte(fmt.Sprintf("%s,%s", req.Method, req.URL.RequestURI())))
	if err != nil {
		return nil, fmt.Errorf("sign request: %w", err)
	}
	req.Header.Set("Authorization", signature)
	return req, nil
}

//export ClaudeDesktopDownloadRequest
func ClaudeDesktopDownloadRequest(authorization **C.char) *C.char {
	if authorization == nil {
		return nil
	}
	*authorization = nil

	req, err := newSignedOllamaRequest(context.Background(), http.MethodGet, claudeDesktopDownloadEndpoint(appui.OllamaDotCom))
	if err != nil {
		slog.Warn("failed to prepare Claude Desktop download request", "error", err)
		return nil
	}
	*authorization = C.CString(req.Header.Get("Authorization"))
	return C.CString(req.URL.String())
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
	claudeProxyStartMu.Lock()
	defer claudeProxyStartMu.Unlock()
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
