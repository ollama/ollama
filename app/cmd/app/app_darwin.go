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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
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
	"golang.org/x/sys/unix"
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
	Running() bool
	Open() error
	ConfigureAutodiscoveryWithAutoMode(autoMode bool) error
	SetInstalledFromDesktopWithAutoMode(installed, restart, autoMode bool) error
	ApplyProfileChange(change func() error, restartConfirmed bool) error
	RestoreForShutdown(ctx context.Context) error
}

var (
	isApp              = updater.BundlePath != ""
	appLogPath         = filepath.Join(os.Getenv("HOME"), ".ollama", "logs", "app.log")
	launchAgentPath    = filepath.Join(os.Getenv("HOME"), "Library", "LaunchAgents", "com.ollama.ollama.plist")
	claudeAppProxy     *proxy.ClaudeDesktop
	claudeProxyStartMu sync.Mutex
	// Serialize default resets with connect, disconnect, and shutdown decisions.
	claudeLifecycleMu sync.Mutex
	claudeProxyMu     sync.Mutex
	claudeCatalogMu   sync.Mutex
	claudeProxyErr    error
	claudeProxyFail   claudeProxyFailure
	claudeDesktop     claudeDesktopController = &launch.ClaudeDesktop{}

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
	claudeCloudModelsClient   = &http.Client{Timeout: 3 * time.Second}
	claudeCloudModelsEndpoint = func() string {
		return strings.TrimRight(appui.OllamaDotCom, "/") + "/api/tags"
	}
	signOllamaData            = ollamaAuth.Sign
	claudeModelsLoader        = loadClaudeDesktopModels
	claudeCloudModelsResolver = currentClaudeDesktopCloudModels
	claudeAvailableModels     []proxy.ClaudeDesktopModel
	claudeModelSource         string
	claudeCatalogUpdated      time.Time

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

type appProcessIdentity struct {
	pid       int
	startedAt int64
}

func (p appProcessIdentity) sameProcess(other appProcessIdentity) bool {
	return p.pid == other.pid && p.startedAt == other.startedAt
}

func (p appProcessIdentity) startedAfter(other appProcessIdentity) bool {
	if p.startedAt != other.startedAt {
		return p.startedAt > other.startedAt
	}
	return p.pid > other.pid
}

type appProcessStopMode uint8

const (
	appProcessStopForHandoff appProcessStopMode = iota
	appProcessStopGracefully
	appProcessStopForcefully
)

type appProcessController struct {
	discover func() ([]appProcessIdentity, error)
	running  func(appProcessIdentity) (bool, error)
	stop     func(appProcessIdentity, appProcessStopMode) error
}

type appSyncBarrierConfig struct {
	handoffTimeout   time.Duration
	terminateTimeout time.Duration
	killTimeout      time.Duration
	pollInterval     time.Duration
	settlePeriod     time.Duration
}

// runAppSyncBarrier elects the newest launch, stops every older instance, and
// returns only after no other instances remain.
func runAppSyncBarrier(self appProcessIdentity, controller appProcessController, config appSyncBarrierConfig) error {
	started := time.Now()
	handoffDeadline := started.Add(config.handoffTimeout)
	terminateDeadline := handoffDeadline.Add(config.terminateTimeout)
	deadline := terminateDeadline.Add(config.killTimeout)
	sawEmpty := false

	for {
		processes, err := controller.discover()
		if err != nil {
			return err
		}
		// Overlapping launches are ordered by process age. Only the newest
		// candidate may terminate existing instances.
		for _, process := range processes {
			if process.startedAfter(self) {
				return fmt.Errorf("%w: pid %d", errNewerAppInstance, process.pid)
			}
		}

		// Require two consecutive empty snapshots so a process that is still
		// appearing in NSWorkspace cannot slip through the barrier.
		if len(processes) == 0 {
			if sawEmpty {
				return nil
			}
			sawEmpty = true
			if !time.Now().Before(deadline) {
				return fmt.Errorf("timed out waiting for app instances to exit")
			}
			time.Sleep(config.settlePeriod)
			continue
		}
		sawEmpty = false

		for _, process := range processes {
			if err := controller.stop(process, appProcessStopForHandoff); err != nil {
				return err
			}
		}
		exited, err := waitForAppProcesses(processes, controller, handoffDeadline, config.pollInterval)
		if err != nil {
			return err
		}
		if !exited {
			for _, process := range processes {
				if err := controller.stop(process, appProcessStopGracefully); err != nil {
					return err
				}
			}
			exited, err = waitForAppProcesses(processes, controller, terminateDeadline, config.pollInterval)
			if err != nil {
				return err
			}
		}
		if !exited {
			// Graceful shutdown owns most of the deadline. Force only exact
			// surviving identities so one stuck instance cannot block update.
			for _, process := range processes {
				if err := controller.stop(process, appProcessStopForcefully); err != nil {
					return err
				}
			}
			exited, err = waitForAppProcesses(processes, controller, deadline, config.pollInterval)
			if err != nil {
				return err
			}
		}
		if !exited {
			return fmt.Errorf("timed out waiting for app instances to exit")
		}
	}
}

func waitForAppProcesses(processes []appProcessIdentity, controller appProcessController, deadline time.Time, pollInterval time.Duration) (bool, error) {
	for {
		running := false
		for _, process := range processes {
			alive, err := controller.running(process)
			if err != nil {
				return false, err
			}
			running = running || alive
		}
		if !running {
			return true, nil
		}
		if !time.Now().Before(deadline) {
			return false, nil
		}
		time.Sleep(pollInterval)
	}
}

const (
	appSyncBarrierHandoffTimeout   = 5 * time.Second
	appSyncBarrierTerminateTimeout = 30 * time.Second
	appSyncBarrierKillTimeout      = 5 * time.Second
	appSyncBarrierPollInterval     = 50 * time.Millisecond
	appSyncBarrierSettlePeriod     = 100 * time.Millisecond
)

var errNewerAppInstance = errors.New("newer app instance owns the handoff")

var killOtherInstances = runDarwinAppSyncBarrier

// Once a replacement handoff starts, later shutdown signals must not restore
// the Claude profile out from under the new app.
var appHandoffInProgress atomic.Bool

func darwinProcessIdentityForPID(pid int) (appProcessIdentity, error) {
	process, err := unix.SysctlKinfoProc("kern.proc.pid", pid)
	if err != nil {
		if errors.Is(err, unix.EIO) && errors.Is(syscall.Kill(pid, 0), syscall.ESRCH) {
			return appProcessIdentity{}, syscall.ESRCH
		}
		return appProcessIdentity{}, err
	}
	if int(process.Proc.P_pid) != pid {
		return appProcessIdentity{}, syscall.ESRCH
	}
	return appProcessIdentity{
		pid:       pid,
		startedAt: process.Proc.P_starttime.Sec*1_000_000 + int64(process.Proc.P_starttime.Usec),
	}, nil
}

func darwinOtherOllamaProcesses() ([]appProcessIdentity, error) {
	var discovered *C.AppProcessIdentity
	var count C.size_t
	if !C.otherOllamaProcesses(&discovered, &count) {
		return nil, errors.New("discover other Ollama app processes")
	}
	defer C.free(unsafe.Pointer(discovered))

	identities := unsafe.Slice(discovered, int(count))
	processes := make([]appProcessIdentity, len(identities))
	for i, process := range identities {
		processes[i] = appProcessIdentity{pid: int(process.pid), startedAt: int64(process.started_at)}
	}
	return processes, nil
}

func darwinAppProcessRunning(expected appProcessIdentity) (bool, error) {
	actual, err := darwinProcessIdentityForPID(expected.pid)
	if errors.Is(err, syscall.ESRCH) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("inspect Ollama app process %d: %w", expected.pid, err)
	}
	return actual.sameProcess(expected), nil
}

func stopDarwinAppProcess(process appProcessIdentity, mode appProcessStopMode) error {
	running, err := darwinAppProcessRunning(process)
	if err != nil || !running {
		return err
	}
	processSignal := syscall.SIGUSR1
	switch mode {
	case appProcessStopGracefully:
		processSignal = syscall.SIGTERM
	case appProcessStopForcefully:
		processSignal = syscall.SIGKILL
	}
	slog.Info("signaling Ollama app process", "pid", process.pid, "signal", processSignal)
	if err := syscall.Kill(process.pid, processSignal); err != nil && !errors.Is(err, syscall.ESRCH) {
		return fmt.Errorf("signal Ollama app process %d: %w", process.pid, err)
	}
	return nil
}

func runDarwinAppSyncBarrier() bool {
	// NSWorkspace snapshots are not atomic, so two concurrent launches can both
	// pass if neither is visible yet. This edge case is intentionally unhandled.
	self, err := darwinProcessIdentityForPID(os.Getpid())
	if err == nil {
		err = runAppSyncBarrier(self, appProcessController{
			discover: darwinOtherOllamaProcesses,
			running:  darwinAppProcessRunning,
			stop:     stopDarwinAppProcess,
		}, appSyncBarrierConfig{
			handoffTimeout:   appSyncBarrierHandoffTimeout,
			terminateTimeout: appSyncBarrierTerminateTimeout,
			killTimeout:      appSyncBarrierKillTimeout,
			pollInterval:     appSyncBarrierPollInterval,
			settlePeriod:     appSyncBarrierSettlePeriod,
		})
	}
	switch {
	case errors.Is(err, errNewerAppInstance):
		slog.Info("newer Ollama app instance owns the handoff")
	case err != nil:
		slog.Warn("app instance sync barrier failed, continuing startup", "error", err)
	}
	return continueAfterBarrierError(err)
}

// continueAfterBarrierError reports whether startup may proceed after the sync
// barrier. Losing the election to a newer instance is the only reason to block
// launch; any other failure leaves at most a stale instance running, so the
// app warns and continues rather than refusing to start.
func continueAfterBarrierError(err error) bool {
	return err == nil || !errors.Is(err, errNewerAppInstance)
}

// handleExistingInstance handles existing instances on macOS.
func handleExistingInstance(_ bool) bool {
	if !isApp {
		return true
	}
	return killOtherInstances()
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
			quitForHandoff()
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
	savedMappings := launch.ClaudeDesktopModelMappings()
	if len(selectedNames) > 0 {
		localNames, err := claudeLocalModelsResolver(ctx)
		if err == nil && allClaudeDesktopModelsLocal(selectedNames, localNames) {
			if len(savedMappings) > 0 {
				selected = proxy.MapClaudeDesktopModels(nil, savedMappings)
			} else {
				selected = proxy.SelectClaudeDesktopModels(nil, selectedNames)
			}
			return selected, selected, "user"
		}
	}
	available, source = claudeModelsLoader(ctx)
	selectable := available
	state, err := claudeAccessStateResolver(ctx)
	if err == nil && state.Cloud == proxy.ClaudeDesktopCloudOn {
		cloudModels, err := claudeCloudModelsResolver(ctx)
		if err != nil {
			slog.Debug("could not load account cloud models for Claude startup", "error", err)
		} else {
			available = mergeClaudeDesktopCloudInventory(available, cloudModels)
			selectable = available
		}
	}
	if len(savedMappings) > 0 {
		selected = proxy.MapClaudeDesktopModels(selectable, savedMappings)
	} else if len(selectedNames) == 0 {
		selected = proxy.MapClaudeDesktopModels(
			selectable,
			proxy.DefaultClaudeDesktopMappingsForModels(selectable),
		)
	} else {
		selected = proxy.SelectClaudeDesktopModels(selectable, selectedNames)
	}
	if len(selected) == 0 && len(selectedNames) > 0 {
		selected = available
	}
	available = includeSelectedClaudeDesktopModels(available, selected)
	if len(selectedNames) > 0 || len(savedMappings) > 0 {
		source = "user"
	}
	return available, selected, source
}

func hasExplicitCloudClaudeDesktopModelName(names []string) bool {
	for _, name := range names {
		if modelref.HasExplicitCloudSource(name) {
			return true
		}
	}
	return false
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
	selected = configuredClaudeDesktopModels(available, nil)
	if len(selected) == 0 {
		selected = available
	}
	available = includeSelectedClaudeDesktopModels(available, selected)
	if len(launch.ClaudeDesktopModels()) > 0 {
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
	var cloudInventory []proxy.ClaudeDesktopModel
	cloudInventoryKnown := false
	if reloaded {
		available, source = claudeModelsLoader(ctx)
		if source == "fallback" {
			if len(previous) > 0 {
				available = proxy.PreserveClaudeDesktopCloudEntitlements(available, previous)
			}
			available = proxy.WithoutClaudeDesktopRecommendationMappings(available)
			current = proxy.WithoutClaudeDesktopRecommendationMappings(current)
		}
		state, err := claudeAccessStateResolver(ctx)
		if err == nil && state.Cloud == proxy.ClaudeDesktopCloudOn {
			cloudModels, err := claudeCloudModelsResolver(ctx)
			if err != nil {
				slog.Debug("could not refresh account cloud models for Claude", "error", err)
			} else {
				cloudInventory = cloudModels
				cloudInventoryKnown = true
				available = mergeClaudeDesktopCloudInventory(available, cloudInventory)
			}
		}
	}
	// Preserve installed local choices. Preserve cloud choices only when the
	// refreshed account inventory still contains their exact Ollama route.
	for _, model := range current {
		if !model.Cloud {
			available = includeSelectedClaudeDesktopModels(available, []proxy.ClaudeDesktopModel{model})
			continue
		}
		if cloudInventoryKnown {
			for _, accountModel := range cloudInventory {
				if accountModel.Name == model.Name || accountModel.OllamaModel == model.OllamaModel {
					available = mergeClaudeDesktopCloudInventory(available, []proxy.ClaudeDesktopModel{accountModel})
					break
				}
			}
		}
	}
	selected = configuredClaudeDesktopModels(available, current)
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

func configuredClaudeDesktopModels(available, current []proxy.ClaudeDesktopModel) []proxy.ClaudeDesktopModel {
	if len(current) > 0 {
		if mappings := proxy.ClaudeDesktopMappings(current); len(mappings) > 0 {
			return proxy.MapClaudeDesktopModels(available, mappings)
		}
	}
	if mappings := launch.ClaudeDesktopModelMappings(); len(mappings) > 0 {
		return proxy.MapClaudeDesktopModels(available, mappings)
	}
	return proxy.SelectClaudeDesktopModels(available, launch.ClaudeDesktopModels())
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

func currentClaudeDesktopCloudModels(ctx context.Context) ([]proxy.ClaudeDesktopModel, error) {
	req, err := newSignedOllamaRequest(ctx, http.MethodGet, claudeCloudModelsEndpoint())
	if err != nil {
		return nil, fmt.Errorf("prepare account cloud model request: %w", err)
	}
	resp, err := claudeCloudModelsClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetch account cloud models: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20))
		return nil, fmt.Errorf("fetch account cloud models: status %d", resp.StatusCode)
	}

	var payload api.ListResponse
	decoder := json.NewDecoder(io.LimitReader(resp.Body, (1<<20)+1))
	if err := decoder.Decode(&payload); err != nil {
		return nil, fmt.Errorf("decode account cloud models: %w", err)
	}
	names := make([]string, 0, len(payload.Models)*2)
	for _, model := range payload.Models {
		if model.Name != "" {
			names = append(names, model.Name)
		}
		if model.Model != "" && model.Model != model.Name {
			names = append(names, model.Model)
		}
	}
	return proxy.ClaudeDesktopModelsFromCloudInventory(names), nil
}

func fallbackClaudeDesktopModels() []proxy.ClaudeDesktopModel {
	models := proxy.UnverifyClaudeDesktopCloudEntitlements(proxy.DefaultClaudeDesktopModels())
	for i := range models {
		models[i].Recommended = false
	}
	return models
}

func claudeDesktopModelSupportsAutoMode(model proxy.ClaudeDesktopModel) bool {
	return model.AccountCloud
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

func mergeClaudeDesktopCloudInventory(available, cloudModels []proxy.ClaudeDesktopModel) []proxy.ClaudeDesktopModel {
	models := proxy.VerifyClaudeDesktopModelsWithCloudInventory(available, cloudModels)
	seen := make(map[string]struct{}, len(models))
	for _, model := range models {
		seen[model.OllamaModel] = struct{}{}
	}
	for _, model := range cloudModels {
		if _, ok := seen[model.OllamaModel]; ok {
			continue
		}
		models = append(models, model)
		seen[model.OllamaModel] = struct{}{}
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
	claudeLifecycleMu.Lock()
	defer claudeLifecycleMu.Unlock()

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
	accessState := proxy.ClaudeDesktopAccessState{
		Cloud:   proxy.ClaudeDesktopCloudUnknown,
		Account: proxy.ClaudeDesktopAccountUnknown,
	}
	if used {
		claudeProxyMu.Lock()
		gateway := claudeAppProxy
		cachedCatalogHasCloud := hasCloudClaudeDesktopModel(claudeAvailableModels)
		claudeProxyMu.Unlock()
		var accessErr error
		accessState, accessErr = claudeAccessStateResolver(context.Background())
		if accessErr != nil {
			slog.Debug("could not resolve Claude model access for Settings", "error", accessErr)
		}
		var current []proxy.ClaudeDesktopModel
		if gateway != nil {
			current = gateway.Models()
		}
		if accessErr == nil && accessState.Cloud == proxy.ClaudeDesktopCloudOn {
			// Local-only startup deliberately seeds the cache with only the active
			// routes. Settings still needs the recommendation catalog when Cloud is on.
			availableModels, selectedModels, modelSource = refreshClaudeDesktopCatalog(context.Background(), current, !cachedCatalogHasCloud)
		} else {
			selectedModels = current
			if len(selectedModels) == 0 {
				selectedModels = proxy.SelectClaudeDesktopModels(nil, launch.ClaudeDesktopModels())
			}
			availableModels = selectedModels
			if len(selectedModels) > 0 {
				modelSource = "user"
			}
		}
	}
	selected := make(map[string]struct{})
	for _, model := range selectedModels {
		selected[model.Name] = struct{}{}
	}
	var localNames []string
	var localErr error
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
	mappedModels := proxy.ClaudeDesktopMappings(selectedModels)
	if len(mappedModels) == 0 && len(launch.ClaudeDesktopModels()) == 0 {
		mappedModels = proxy.DefaultClaudeDesktopMappingsForModels(availableModels)
	}
	mappingStatuses := make([]claudeDesktopMappingStatus, 0, proxy.MaxClaudeDesktopModels)
	for _, route := range proxy.ClaudeDesktopRoutes() {
		mappingStatuses = append(mappingStatuses, claudeDesktopMappingStatus{
			RouteID:   route.ID,
			RouteName: route.DisplayName,
			Model:     mappedModels[route.ID],
		})
	}

	status := claudeDesktopConnectionSummary(used)
	autoMode, autoModeErr := launch.ClaudeDesktopAutoModeEnabled()
	status.AutoMode = autoMode
	status.ModelSource = modelSource
	status.Models = modelStatuses
	status.Mappings = mappingStatuses
	if status.Error == "" && autoModeErr != nil {
		status.Error = autoModeErr.Error()
	}
	return status
}

func resolveClaudeDesktopDefaultMappings(ctx context.Context) (map[string]string, error) {
	available, _ := claudeModelsLoader(ctx)
	mappings := proxy.DefaultClaudeDesktopMappingsForModels(available)
	if len(mappings) == 0 {
		return nil, errors.New("no Claude model mapping defaults are available")
	}
	return mappings, nil
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

func setClaudeDesktopAutoMode(enabled, restartConfirmed bool) error {
	models := activeClaudeDesktopModels()
	if enabled && !claudeDesktopModelsSupportAutoMode(models) {
		return errors.New("select at least one cloud model available to your Ollama.com account")
	}
	previous, err := launch.ClaudeDesktopAutoModeEnabled()
	if err != nil {
		return err
	}
	if previous == enabled && (!claudeDesktop.UsesOllamaGateway() || claudeDesktop.AutodiscoveryConfiguredWithAutoMode(enabled)) {
		return nil
	}
	if !claudeDesktop.UsesOllamaGateway() {
		// The preference takes effect the next time the profile is written.
		if err := launch.SaveClaudeDesktopAutoMode(enabled); err != nil {
			return fmt.Errorf("save Claude Desktop auto mode: %w", err)
		}
		return nil
	}
	return claudeDesktop.ApplyProfileChange(func() error {
		// Persist only after the native layer has established that a running
		// Claude process may be restarted. Canceling consent must be a no-op.
		if err := launch.SaveClaudeDesktopAutoMode(enabled); err != nil {
			return fmt.Errorf("save Claude Desktop auto mode: %w", err)
		}
		return claudeDesktop.ConfigureAutodiscoveryWithAutoMode(enabled)
	}, restartConfirmed)
}

func applyClaudeDesktopMappings(mappings map[string]string, restartConfirmed bool) (bool, error) {
	return applyClaudeDesktopMappingsWithOpen(mappings, restartConfirmed, true)
}

func resetClaudeDesktopMappings(restartConfirmed bool) (bool, error) {
	claudeLifecycleMu.Lock()
	defer claudeLifecycleMu.Unlock()

	if !hasUsedClaudeDesktopIntegration() {
		return false, nil
	}
	mappings, err := resolveClaudeDesktopDefaultMappings(context.Background())
	if err != nil {
		return false, err
	}
	if !claudeDesktop.UsesOllamaGateway() {
		if maps.Equal(launch.ClaudeDesktopModelMappings(), mappings) {
			return false, nil
		}
		if err := launch.SaveClaudeDesktopModelMappings(mappings); err != nil {
			return false, fmt.Errorf("save Claude Desktop model mappings: %w", err)
		}
		return true, nil
	}
	return applyClaudeDesktopMappingsWithOpen(mappings, restartConfirmed, false)
}

func applyClaudeDesktopMappingsWithOpen(mappings map[string]string, restartConfirmed, openWhenStopped bool) (bool, error) {
	if !claudeDesktopInstalled() {
		return false, errors.New("Claude Desktop is not installed")
	}
	knownRoutes := make(map[string]struct{}, proxy.MaxClaudeDesktopModels)
	for _, route := range proxy.ClaudeDesktopRoutes() {
		knownRoutes[route.ID] = struct{}{}
	}
	for routeID, model := range mappings {
		if _, ok := knownRoutes[routeID]; !ok {
			return false, fmt.Errorf("unknown Claude Desktop route %q", routeID)
		}
		if strings.TrimSpace(model) == "" {
			continue
		}
	}
	if len(mappings) == 0 {
		return false, errors.New("map at least one Claude Desktop route")
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
	accessState, accessErr := claudeAccessStateResolver(context.Background())
	if accessErr != nil {
		slog.Debug("could not resolve Claude model access for selection", "error", accessErr)
	}
	selectable := available
	mappedNames := make([]string, 0, len(mappings))
	for _, name := range mappings {
		if name = strings.TrimSpace(name); name != "" {
			mappedNames = append(mappedNames, name)
		}
	}
	if accessErr == nil && accessState.Cloud == proxy.ClaudeDesktopCloudOn && hasExplicitCloudClaudeDesktopModelName(mappedNames) {
		cloudModels, err := claudeCloudModelsResolver(context.Background())
		if err != nil {
			slog.Debug("could not load account cloud models for Claude selection", "error", err)
		} else {
			selectable = mergeClaudeDesktopCloudInventory(available, cloudModels)
		}
	}
	selected, err := mapKnownClaudeDesktopModels(selectable, current, localNames, mappings)
	if err != nil {
		return false, err
	}
	if err := ensureClaudeDesktopModelsAvailable(context.Background(), selected); err != nil {
		return false, err
	}

	normalized := proxy.ClaudeDesktopMappings(selected)
	previousSelection := launch.ClaudeDesktopModels()
	previousMappings := launch.ClaudeDesktopModelMappings()
	mappingsChanged := !maps.Equal(previousMappings, normalized)
	// The Settings button must never restart a live Claude process when there
	// is no mapping change to apply. A stopped app may still use the same button
	// to repair its profile and launch Claude.
	if !mappingsChanged && claudeDesktop.Running() {
		return false, nil
	}
	restoreState := func() error {
		var rollbackErr error
		if gateway != nil && len(current) > 0 {
			rollbackErr = gateway.SetModels(current)
		}
		if err := launch.RestoreClaudeDesktopModelMappings(previousSelection, previousMappings); err != nil {
			rollbackErr = errors.Join(rollbackErr, fmt.Errorf("restore Claude Desktop model mappings: %w", err))
		}
		claudeProxyMu.Lock()
		claudeAvailableModels = previousAvailable
		claudeModelSource = previousSource
		claudeCatalogUpdated = previousCatalogUpdated
		claudeProxyMu.Unlock()
		return rollbackErr
	}
	if gateway == nil {
		applyInitialChange := func() error {
			if mappingsChanged {
				if err := launch.SaveClaudeDesktopModelMappings(normalized); err != nil {
					return fmt.Errorf("save Claude Desktop model mappings: %w", err)
				}
			}
			if err := startClaudeAppProxy(); err != nil {
				return err
			}
			autoMode, err := effectiveClaudeDesktopAutoMode(selected)
			if err != nil {
				return err
			}
			return claudeDesktop.ConfigureAutodiscoveryWithAutoMode(autoMode)
		}
		if err := claudeDesktop.ApplyProfileChange(applyInitialChange, restartConfirmed); err != nil {
			if errors.Is(err, launch.ErrClaudeDesktopRestartConfirmationRequired) {
				return false, err
			}
			stopClaudeAppProxy()
			return false, errors.Join(err, restoreState())
		}
		if openWhenStopped && !claudeDesktop.Running() {
			if err := claudeDesktop.Open(); err != nil {
				if mappingsChanged {
					return true, fmt.Errorf("Claude model mappings were saved, but Claude Desktop could not open: %w", err)
				}
				return false, fmt.Errorf("open Claude Desktop: %w", err)
			}
		}
		return mappingsChanged, nil
	}
	applyModelChange := func() error {
		if mappingsChanged {
			if err := launch.SaveClaudeDesktopModelMappings(normalized); err != nil {
				return fmt.Errorf("save Claude Desktop model mappings: %w", err)
			}
			if err := gateway.SetModels(selected); err != nil {
				return err
			}
			claudeProxyMu.Lock()
			claudeAvailableModels = includeSelectedClaudeDesktopModels(available, selected)
			claudeModelSource = "user"
			claudeProxyMu.Unlock()
		}
		autoMode, err := effectiveClaudeDesktopAutoMode(selected)
		if err != nil {
			return err
		}
		if err := claudeDesktop.ConfigureAutodiscoveryWithAutoMode(autoMode); err != nil {
			return err
		}
		return nil
	}
	if err := claudeDesktop.ApplyProfileChange(applyModelChange, restartConfirmed); err != nil {
		if errors.Is(err, launch.ErrClaudeDesktopRestartConfirmationRequired) {
			return false, err
		}
		return false, errors.Join(err, restoreState())
	}
	if openWhenStopped && !claudeDesktop.Running() {
		if err := claudeDesktop.Open(); err != nil {
			if mappingsChanged {
				return true, fmt.Errorf("Claude model mappings were saved, but Claude Desktop could not open: %w", err)
			}
			return false, fmt.Errorf("open Claude Desktop: %w", err)
		}
	}
	return mappingsChanged, nil
}

func mapKnownClaudeDesktopModels(available, current []proxy.ClaudeDesktopModel, localNames []string, mappings map[string]string) ([]proxy.ClaudeDesktopModel, error) {
	selectable := includeSelectedClaudeDesktopModels(available, current)
	allowed := make(map[string]struct{}, len(selectable)+len(localNames))
	for _, model := range selectable {
		allowed[model.Name] = struct{}{}
		allowed[model.OllamaModel] = struct{}{}
	}
	for _, name := range localNames {
		allowed[strings.TrimSpace(name)] = struct{}{}
	}
	for routeID, rawName := range mappings {
		name := strings.TrimSpace(rawName)
		if name == "" {
			continue
		}
		if _, ok := allowed[name]; !ok {
			return nil, fmt.Errorf("model %q mapped from %s is not installed or recommended for Claude Desktop", name, routeID)
		}
	}

	selected := proxy.MapClaudeDesktopModels(selectable, mappings)
	if len(selected) == 0 {
		return nil, errors.New("map at least one Claude Desktop route")
	}
	return selected, nil
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
			return nil, fmt.Errorf("model %q is not installed, recommended, or available to this Ollama.com account", name)
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

func quitForHandoff() {
	appHandoffInProgress.Store(true)
	quit()
}

func quit() {
	ctx, cancel := context.WithTimeout(context.Background(), claudeShutdownTimeout)
	defer cancel()
	if err := restoreClaudeAppForTermination(ctx, appHandoffInProgress.Load()); err != nil {
		slog.Warn("failed to restore Claude before quitting", "error", err)
	}
	C.quit()
}

func restoreClaudeBeforeQuit(ctx context.Context, configured bool, restore func(context.Context) error) error {
	if !configured {
		return nil
	}
	return restore(ctx)
}

func restoreClaudeAppForTermination(ctx context.Context, handoff bool) error {
	claudeLifecycleMu.Lock()
	defer claudeLifecycleMu.Unlock()

	if handoff {
		stopClaudeAppProxy()
		return nil
	}
	configured := claudeDesktop.UsesOllamaGateway()
	err := restoreClaudeBeforeQuit(ctx, configured, claudeDesktop.RestoreForShutdown)
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
